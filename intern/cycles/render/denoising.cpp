#include "render/denoising.h"

#include "util/util_image.h"

CCL_NAMESPACE_BEGIN

typedef enum DenoisingPassType {
	DENOISING_PASS_READ_NONE                      = 0,
	DENOISING_PASS_READ_NORMAL            = (1 << 0),
	DENOISING_PASS_READ_NORMAL_VAR        = (1 << 1),
	DENOISING_PASS_READ_ALBEDO            = (1 << 2),
	DENOISING_PASS_READ_ALBEDO_VAR        = (1 << 3),
	DENOISING_PASS_READ_DEPTH             = (1 << 4),
	DENOISING_PASS_READ_DEPTH_VAR         = (1 << 5),
	DENOISING_PASS_READ_SHADOW_A          = (1 << 6),
	DENOISING_PASS_READ_SHADOW_B          = (1 << 7),
	DENOISING_PASS_READ_IMAGE             = (1 << 8),
	DENOISING_PASS_READ_IMAGE_VAR         = (1 << 9),
	DENOISING_PASS_READ_CLEAN             = (1 << 10),
	DENOISING_PASS_READ_IMAGE_B           = (1 << 11),
	DENOISING_PASS_READ_IMAGE_B_VAR       = (1 << 12),
	
	DENOISING_PASS_READ_REQUIRED = (DENOISING_PASS_READ_NORMAL
							   | DENOISING_PASS_READ_NORMAL_VAR
							   | DENOISING_PASS_READ_ALBEDO
							   | DENOISING_PASS_READ_ALBEDO_VAR
							   | DENOISING_PASS_READ_DEPTH
							   | DENOISING_PASS_READ_DEPTH_VAR
							   | DENOISING_PASS_READ_SHADOW_A
							   | DENOISING_PASS_READ_SHADOW_B
							   | DENOISING_PASS_READ_IMAGE
							   | DENOISING_PASS_READ_IMAGE_VAR),
	DENOISING_PASS_READ_ALL = DENOISING_PASS_READ_REQUIRED | DENOISING_PASS_READ_CLEAN | DENOISING_PASS_READ_IMAGE_B | DENOISING_PASS_READ_IMAGE_B_VAR,
} DenoisingPassType;

typedef class PassTypeInfo
{
public:
	PassTypeInfo(DenoisingPassType type, int num_channels, string channels)
	 : type(type), num_channels(num_channels), channels(channels) {}
	PassTypeInfo() : type(DENOISING_PASS_READ_NONE), num_channels(0), channels("") {}

	DenoisingPassType type;
	int num_channels;
	string channels;

	bool operator<(const PassTypeInfo &other) const {
		return type < other.type;
	}
} PassTypeInfo;

static map<string, PassTypeInfo> denoise_passes_init()
{
	map<string, PassTypeInfo> passes;

	passes["Denoising Normal"]          = PassTypeInfo(DENOISING_PASS_READ_NORMAL,     3, "XYZ");
	passes["Denoising Normal Variance"] = PassTypeInfo(DENOISING_PASS_READ_NORMAL_VAR, 3, "XYZ");
	passes["Denoising Albedo"]          = PassTypeInfo(DENOISING_PASS_READ_ALBEDO,     3, "RGB");
	passes["Denoising Albedo Variance"] = PassTypeInfo(DENOISING_PASS_READ_ALBEDO_VAR, 3, "RGB");
	passes["Denoising Depth"]           = PassTypeInfo(DENOISING_PASS_READ_DEPTH,      1, "Z");
	passes["Denoising Depth Variance"]  = PassTypeInfo(DENOISING_PASS_READ_DEPTH_VAR,  1, "Z");
	passes["Denoising Shadow A"]        = PassTypeInfo(DENOISING_PASS_READ_SHADOW_A,   3, "XYV");
	passes["Denoising Shadow B"]        = PassTypeInfo(DENOISING_PASS_READ_SHADOW_B,   3, "XYV");
	passes["Denoising Image"]           = PassTypeInfo(DENOISING_PASS_READ_IMAGE,      3, "RGB");
	passes["Denoising Image Variance"]  = PassTypeInfo(DENOISING_PASS_READ_IMAGE_VAR,  3, "RGB");
	passes["Denoising Clean"]           = PassTypeInfo(DENOISING_PASS_READ_CLEAN,      3, "RGB");

	return passes;
}

static map<string, PassTypeInfo> denoise_passes_map = denoise_passes_init();

static bool split_channel(string full_channel, string &layer, string &pass, string &channel)
{
	/* Splits channel name into <layer>.<pass>.<channel> */
	if(std::count(full_channel.begin(), full_channel.end(), '.') != 2) {
		return false;
	}

	int first_dot = full_channel.find(".");
	int second_dot = full_channel.rfind(".");
	layer = full_channel.substr(0, first_dot);
	pass = full_channel.substr(first_dot + 1, second_dot - first_dot - 1);
	channel = full_channel.substr(second_dot + 1);

	return true;
}

static int find_channel(string channels, string channel)
{
	if(channel.length() != 1) return -1;
	size_t pos = channels.find(channel);
	if(pos == string::npos) return -1;
	return pos;
}

static RenderBuffers* load_frame(string file, Device *device, RenderBuffers *buffers, int samples, int numframes, int framenum)
{
	ImageInput *frame = ImageInput::open(file);
	if(!frame) {
		printf("ERROR: Frame %s: Couldn't open file!\n", file.c_str());
		delete buffers;
		return NULL;
	}

	const ImageSpec &spec = frame->spec();

	if(buffers) {
		if(spec.width != buffers->params.width || spec.height != buffers->params.height) {
			printf("ERROR: Frame %s: Has different size!\n", file.c_str());
			delete buffers;
			return NULL;
		}
	}

	/* Find a single RenderLayer to load. */
	string renderlayer = "";
	string layer, pass, channel;
	for(int i = 0; i < spec.nchannels; i++) {
		if(!split_channel(spec.channelnames[i], layer, pass, channel)) continue;
		if(pass == "Denoising Image") {
			renderlayer = layer;
			break;
		}
	}

	if(renderlayer != "") {
		/* Find all passes that the frame contains. */
		int passes = DENOISING_PASS_READ_NONE;
		map<DenoisingPassType, int> num_channels;
		map<PassTypeInfo, int3> channel_ids;
		for(int i = 0; i < spec.nchannels; i++) {
			if(!split_channel(spec.channelnames[i], layer, pass, channel)) continue;
			if(layer != renderlayer) {
				/* The channel belongs to another RenderLayer. */
				continue;
			}
			if(denoise_passes_map.count(pass)) {
				PassTypeInfo type = denoise_passes_map[pass];
				assert(type.num_channels <= 3);
				/* Pass was found, count the channels. */
				size_t channel_id = find_channel(type.channels, channel);
				if(channel_id != -1) {
					/* This channel is part of the pass, so count it. */
					num_channels[type.type]++;
					/* Remember which OIIO channel belongs to which pass. */
					channel_ids[type][channel_id] = i;
					if(num_channels[type.type] == type.num_channels) {
						/* We found all the channels of the pass! */
						passes |= type.type;
					}
				}
			}
		}

		/* The frame always needs to include all the required denoising passes.
		 * If the primary frame also included a clean pass, all the secondary frames need to do so as well. */
		if((~passes & DENOISING_PASS_READ_REQUIRED) == 0 && !(buffers && buffers->params.denoising_clean_pass && !(passes & DENOISING_PASS_READ_CLEAN))) {
			printf("Frame %s: Found all needed passes!\n", file.c_str());

			if(buffers == NULL) {
				BufferParams params;
				params.width  = params.full_width  /* = params.final_width */  = spec.width;
				params.height = params.full_height/* = params.final_height*/ = spec.height;
				params.full_x = params.full_y = 0;
				params.denoising_data_pass = true;
				params.denoising_clean_pass = (passes & DENOISING_PASS_READ_CLEAN);
				params.cross_denoising = (passes & DENOISING_PASS_READ_IMAGE_B) && (passes & DENOISING_PASS_READ_IMAGE_B_VAR);
				params.frames = numframes;

				buffers = new RenderBuffers(device);
				buffers->reset(device, params);
			}

			int4 rect = make_int4(0, 0, buffers->params.width, buffers->params.height);
			float *pass_data = new float[4*buffers->params.width*buffers->params.height];

			/* Read all the passes from the file. */
			for(map<PassTypeInfo, int3>::iterator i = channel_ids.begin(); i != channel_ids.end(); i++)
			{
				for(int c = 0; c < i->first.num_channels; c++) {
					int xstride = i->first.num_channels*sizeof(float);
					int ystride = spec.width * xstride;
					printf("Reading pass %s!            \r", spec.channelnames[i->second[c]].c_str());
					fflush(stdout);
					frame->read_image(i->second[c], i->second[c]+1, TypeDesc::FLOAT, pass_data + c, xstride, ystride);
				}
				
				float exposure = 1.0f;
				int type_offset = 0;
				switch(i->first.type) {
					case DENOISING_PASS_READ_NONE:        assert(0); break;
					case DENOISING_PASS_READ_NORMAL:      type_offset = DENOISING_PASS_NORMAL; break;
					case DENOISING_PASS_READ_NORMAL_VAR:  type_offset = DENOISING_PASS_NORMAL_VAR; break;
					case DENOISING_PASS_READ_ALBEDO:      type_offset = DENOISING_PASS_ALBEDO; break;
					case DENOISING_PASS_READ_ALBEDO_VAR:  type_offset = DENOISING_PASS_ALBEDO_VAR; break;
					case DENOISING_PASS_READ_DEPTH:       type_offset = DENOISING_PASS_DEPTH; break;
					case DENOISING_PASS_READ_DEPTH_VAR:   type_offset = DENOISING_PASS_DEPTH_VAR; break;
					case DENOISING_PASS_READ_SHADOW_A:    type_offset = DENOISING_PASS_SHADOW_A; break;
					case DENOISING_PASS_READ_SHADOW_B:    type_offset = DENOISING_PASS_SHADOW_B; break;
					case DENOISING_PASS_READ_IMAGE:       type_offset = DENOISING_PASS_COLOR; break;
					case DENOISING_PASS_READ_IMAGE_VAR:   type_offset = DENOISING_PASS_COLOR_VAR; break;
					case DENOISING_PASS_READ_IMAGE_B:     continue; break;
					case DENOISING_PASS_READ_IMAGE_B_VAR: continue; break;
					case DENOISING_PASS_READ_CLEAN:       continue; break;
				}
				
				buffers->set_denoising_pass_rect(type_offset, exposure, samples, i->first.num_channels, rect, pass_data, framenum);
			}

			/* Read combined pass. */
			int read_combined = 0;
			for(int i = 0; i < spec.nchannels; i++) {
				if(!split_channel(spec.channelnames[i], layer, pass, channel)) continue;
				if(layer != renderlayer || pass != "Combined") continue;

				size_t channel_id = find_channel("RGBA", channel);
				if(channel_id != -1) {
					int xstride = 4*sizeof(float);
					int ystride = spec.width * xstride;
					printf("Reading pass %s!            \n", spec.channelnames[i].c_str());
					fflush(stdout);
					frame->read_image(i, i+1, TypeDesc::FLOAT, pass_data + channel_id, xstride, ystride);
					read_combined++;
				}
			}
			if(read_combined < 4) {
				printf("ERROR: Frame %s: Missing combined pass!\n", file.c_str());
				delete buffers;
				delete[] pass_data;
				return NULL;
			}

			buffers->set_combined_pass_rect(rect, pass_data, framenum, samples);
			//buffers->get_pass_rect(PASS_COMBINED, 1.0f, samples, 4, rect, pass_data, true, framenum);

			delete[] pass_data;
		}
		else {
			printf("ERROR: Frame %s: Missing some pass!\n", file.c_str());
			delete buffers;
			return NULL;
		}
	}
	else {
		printf("ERROR: Frame %s: Didn't fine a suitable RenderLayer!\n", file.c_str());
		delete buffers;
		return NULL;
	}

	frame->close();
	ImageInput::destroy(frame);

	return buffers;
}

bool denoise_standalone(SessionParams &session_params,
                        vector<string> &frames,
                        int mid_frame)
{
	session_params.use_denoising = true;
	session_params.only_denoise = true;
	session_params.progressive_refine = false;
	session_params.progressive = false;
	session_params.background = true;
	session_params.tile_order = TILE_BOTTOM_TO_TOP;
//	session_params.flip_output = false;
//	session_params.prev_frames = mid_frame;

	Session *session = new Session(session_params);
	session->set_pause(false);

	int framenum = 0;
	RenderBuffers *buffers = load_frame(frames[mid_frame], session->device, NULL, session_params.samples, frames.size(), framenum++);
	if(buffers == NULL) {
		delete session;
		return false;
	}
	for(int i = 0; i < frames.size(); i++) {
		if(i == mid_frame) continue;
		buffers = load_frame(frames[i], session->device, buffers, session_params.samples, frames.size(), framenum++);
		if(buffers == NULL) {
			delete session;
			return false;
		}
	}

	buffers->copy_to_device();
	session->buffers = buffers;
	session->params.filter_cross = buffers->params.cross_denoising;

	session->start_denoise();
	session->wait();

	/* Required for correct scaling of the output. */
	session->params.samples--;

	delete session;

	return true;
}

CCL_NAMESPACE_END
