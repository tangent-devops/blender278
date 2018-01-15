/*
 * ***** BEGIN GPL LICENSE BLOCK *****
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
 * along with this program; if not, write to the Free Software  Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2005 by the Blender Foundation.
 * All rights reserved.
 *
 * Contributor(s): Your name
 *
 * ***** END GPL LICENSE BLOCK *****
 *
 */
 
/** \file blender/modifiers/intern/MOD_scaling.c
 *  \ingroup modifiers
 */
 
 
 #include "DNA_meshdata_types.h"
 
#include "BLI_math.h"
#include "BLI_utildefines.h"
#include "BLI_string.h"
 
#include "MEM_guardedalloc.h"
 
#include "BKE_cdderivedmesh.h"
#include "BKE_particle.h"
#include "BKE_deform.h"
 
#include "MOD_modifiertypes.h"
#include "MOD_util.h"
 
 
static void initData(ModifierData *md)
{
	ScalingModifierData *smd = (ScalingModifierData *) md;
	smd->scale = 1.0f;
}
 
static void copyData(ModifierData *md, ModifierData *target)
{
	ScalingModifierData *smd = (ScalingModifierData *) md;
	ScalingModifierData *tsmd = (ScalingModifierData *) target;
	tsmd->scale = smd->scale;
}
 
static int isDisabled(ModifierData *md, int UNUSED(useRenderParams))
{
	ScalingModifierData *smd = (ScalingModifierData *) md;
	/* disable if modifier is 1.0 for scale*/
	if (smd->scale == 1.0f) return 1;
	return 0;
}
 
static CustomDataMask requiredDataMask(Object *UNUSED(ob), ModifierData *md)
{
	ScalingModifierData *smd = (ScalingModifierData *)md;
	CustomDataMask dataMask = 0;
	return dataMask;
}
 
static void ScalingModifier_do(
        ScalingModifierData *smd, Object *ob, DerivedMesh *dm,
        float (*vertexCos)[3], int numVerts)
{
	int i;
	float scale;
	scale = smd->scale;
 
	for (i = 0; i < numVerts; i++) {
		vertexCos[i][0] = vertexCos[i][0] * scale;
		vertexCos[i][1] = vertexCos[i][1] * scale;
		vertexCos[i][2] = vertexCos[i][2] * scale;
	}
}
 
static void deformVerts(ModifierData *md, Object *ob, DerivedMesh *derivedData,
                        float (*vertexCos)[3], int numVerts, ModifierApplyFlag UNUSED(flag))
{
	DerivedMesh *dm = get_dm(ob, NULL, derivedData, NULL, false, false);
 
	ScalingModifier_do((ScalingModifierData *)md, ob, dm,
	                  vertexCos, numVerts);
 
	if (dm != derivedData)
		dm->release(dm);
}
 
static void deformVertsEM(
        ModifierData *md, Object *ob, struct BMEditMesh *editData,
        DerivedMesh *derivedData, float (*vertexCos)[3], int numVerts)
{
	DerivedMesh *dm = get_dm(ob, editData, derivedData, NULL, false, false );
 
	ScalingModifier_do((ScalingModifierData *)md, ob, dm,
	                  vertexCos, numVerts);
 
	if (dm != derivedData)
		dm->release(dm);
}
 
 
ModifierTypeInfo modifierType_Scaling = {
	/* name */              "Scaling",
	/* structName */        "ScalingModifierData",
	/* structSize */        sizeof(ScalingModifierData),
	/* type */              eModifierTypeType_OnlyDeform,
	/* flags */             eModifierTypeFlag_AcceptsMesh |
	                        eModifierTypeFlag_SupportsEditmode,
 
	/* copyData */          copyData,
	/* deformVerts */       deformVerts,
	/* deformMatrices */    NULL,
	/* deformVertsEM */     deformVertsEM,
	/* deformMatricesEM */  NULL,
	/* applyModifier */     NULL,
	/* applyModifierEM */   NULL,
	/* initData */          initData,
	/* requiredDataMask */  requiredDataMask,
	/* freeData */          NULL,
	/* isDisabled */        isDisabled,
	/* updateDepgraph */    NULL,
	/* dependsOnTime */     NULL,
	/* dependsOnNormals */	NULL,
	/* foreachObjectLink */ NULL,
	/* foreachIDLink */     NULL,
	/* foreachTexLink */    NULL,
};
