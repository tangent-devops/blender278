/*
 * Copyright 2011-2016 Blender Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __KERNEL_CPU_IMAGE_H__
#define __KERNEL_CPU_IMAGE_H__

#ifdef __KERNEL_CPU__

CCL_NAMESPACE_BEGIN

ccl_device float4 kernel_tex_image_interp_impl(KernelGlobals *kg, int tex, float x, float y)
{
	switch(kernel_tex_type(tex)) {
		case IMAGE_DATA_TYPE_BYTE4:
			return kg->texture_byte4_images[kernel_tex_index(tex)].interp(x, y);
		case IMAGE_DATA_TYPE_HALF4:
			return kg->texture_half4_images[kernel_tex_index(tex)].interp(x, y);
		case IMAGE_DATA_TYPE_USHORT4:
			return kg->texture_ushort4_images[kernel_tex_index(tex)].interp(x, y);
		case IMAGE_DATA_TYPE_FLOAT:
			return kg->texture_float_images[kernel_tex_index(tex)].interp(x, y);
		case IMAGE_DATA_TYPE_BYTE:
			return kg->texture_byte_images[kernel_tex_index(tex)].interp(x, y);
		case IMAGE_DATA_TYPE_HALF:
			return kg->texture_half_images[kernel_tex_index(tex)].interp(x, y);
		case IMAGE_DATA_TYPE_USHORT:
			return kg->texture_ushort_images[kernel_tex_index(tex)].interp(x, y);
		case IMAGE_DATA_TYPE_FLOAT4:
		default:
			return kg->texture_float4_images[kernel_tex_index(tex)].interp(x, y);
	}
}

ccl_device float4 kernel_tex_image_interp_3d_impl(KernelGlobals *kg, int tex, float x, float y, float z)
{
	const int index = kernel_tex_index(tex);
	const int type = kernel_tex_type(tex);
#ifdef __OPENVDB__
	if(kg->vdb) {
		if(type == IMAGE_DATA_TYPE_FLOAT) {
			float r = VDBVolume::sample_scalar(kg->vdb, kg->vdb_tdata, index, x, y, z, 1);
			if(r != 0.0f) {
				r = 1.0f;
			}
			return make_float4(r, r, r, 1.0f);
		} else if(type == IMAGE_DATA_TYPE_FLOAT4) {
			float r, g, b;
			VDBVolume::sample_vector(kg->vdb, kg->vdb_tdata, index, x, y, z, &r, &g, &b, 1);
			return make_float4(r, g, b, 1.0f);
		}
	}
#endif
	switch(type) {
		case IMAGE_DATA_TYPE_BYTE4:
			return kg->texture_byte4_images[index].interp_3d(x, y, z);
		case IMAGE_DATA_TYPE_HALF4:
			return kg->texture_half4_images[index].interp_3d(x, y, z);
		case IMAGE_DATA_TYPE_USHORT4:
			return kg->texture_ushort4_images[index].interp_3d(x, y, z);
		case IMAGE_DATA_TYPE_FLOAT:
			return kg->texture_float_images[index].interp_3d(x, y, z);
		case IMAGE_DATA_TYPE_BYTE:
			return kg->texture_byte_images[index].interp_3d(x, y, z);
		case IMAGE_DATA_TYPE_HALF:
			return kg->texture_half_images[index].interp_3d(x, y, z);
		case IMAGE_DATA_TYPE_USHORT:
			return kg->texture_ushort_images[index].interp_3d(x, y, z);
		case IMAGE_DATA_TYPE_FLOAT4:
		default:
			return kg->texture_float4_images[index].interp_3d(x, y, z);
	}
}

ccl_device float4 kernel_tex_image_interp_3d_ex_impl(KernelGlobals *kg, int tex, float x, float y, float z, int interpolation)
{
	switch(kernel_tex_type(tex)) {
		case IMAGE_DATA_TYPE_BYTE4:
			return kg->texture_byte4_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
		case IMAGE_DATA_TYPE_HALF4:
			return kg->texture_half4_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
		case IMAGE_DATA_TYPE_USHORT4:
			return kg->texture_ushort4_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
		case IMAGE_DATA_TYPE_FLOAT:
			return kg->texture_float_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
		case IMAGE_DATA_TYPE_BYTE:
			return kg->texture_byte_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
		case IMAGE_DATA_TYPE_HALF:
			return kg->texture_half_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
		case IMAGE_DATA_TYPE_USHORT:
			return kg->texture_ushort_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
		case IMAGE_DATA_TYPE_FLOAT4:
		default:
			return kg->texture_float4_images[kernel_tex_index(tex)].interp_3d_ex(x, y, z, interpolation);
	}
}

CCL_NAMESPACE_END

#endif  // __KERNEL_CPU__


#endif // __KERNEL_CPU_IMAGE_H__
