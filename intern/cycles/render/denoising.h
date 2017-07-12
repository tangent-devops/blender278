/*
 * Copyright 2011-2014 Blender Foundation
 *
 * Licensed under the Apache License,
 Version 2.0 (the "License");
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

#ifndef __DENOISING_H__
#define __DENOISING_H__

#include "session.h"

CCL_NAMESPACE_BEGIN

bool denoise_standalone(SessionParams &session_params,
                        vector<string> &frames,
                        int mid_frame);

CCL_NAMESPACE_END

#endif /* __DENOISING_H__ */