/*
* Copyright 2011, Blender Foundation.
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software Foundation,
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*
* Contributor:
*		Jeroen Bakker
*		Monique Dewanchand
*		Cristian Kovacs (only multi mix)
*/

#include "COM_MultiMixOperation.h"

extern "C" {
#  include "BLI_math.h"
}

/* ******** Multi Mix Operation ******** */
MultiMixOperation::MultiMixOperation(size_t num_inputs) : NodeOperation()
{
	for (size_t i = 0; i < num_inputs; i++)
		this->addInputSocket(COM_DT_COLOR);
	inputs.resize(num_inputs);
	this->addOutputSocket(COM_DT_COLOR);
	this->setUseClamp(false);
}

void MultiMixOperation::initExecution()
{
	for (size_t i = 0; i < inputs.size(); i++)
		inputs[i] = this->getInputSocketReader(i);
}

void MultiMixOperation::executePixelSampled(float output[4], float x, float y, PixelSampler sampler)
{
	int arrsize = inputs.size();
	float inputValue[4];
	float inputColor[4];

	inputs[0]->readSampled(inputColor, x, y, sampler);

	output[0] = output[1] = output[2] = 0;

	float value = inputColor[0];

	for (size_t i = 1; i <= inputs.size()-1; i++) {
		inputs[i]->readSampled(inputColor, x, y, sampler);
		if (i == 1)
			output[3] = inputColor[3];
		output[0] += inputColor[0]*powf(value, ceil(((float)(i-1))/i))*powf(1.0-value, (arrsize-1)-i);
		output[1] += inputColor[1]*powf(value, ceil(((float)(i-1))/i))*powf(1.0-value, (arrsize-1)-i);
		output[2] += inputColor[2]*powf(value, ceil(((float)(i-1))/i))*powf(1.0-value, (arrsize-1)-i);
	}

	clampIfNeeded(output);
}

void MultiMixOperation::determineResolution(unsigned int resolution[2], unsigned int preferredResolution[2])
{
	NodeOperationInput *socket;
	unsigned int tempPreferredResolution[2] = { 0, 0 };
	unsigned int tempResolution[2];

	for (size_t i = 0; i < inputs.size(); i++) {
		socket = this->getInputSocket(i);
		socket->determineResolution(tempResolution, tempPreferredResolution);
		if ((tempResolution[0] != 0) && (tempResolution[1] != 0)) {
			this->setResolutionInputSocketIndex(i);
			break;
		}
	}
	NodeOperation::determineResolution(resolution, preferredResolution);
}

void MultiMixOperation::deinitExecution()
{
	for (size_t i = 0; i < inputs.size(); i++)
		inputs[i] = NULL;
}