#!/usr/bin/env python
import os
import sys

import pipebuilder as pb
from pipebuilder import tracking
import pipebuilder.custom

import pdb
import copy

from stroke_processing.tools import config
cwd = os.path.dirname(os.path.abspath(__file__))

CLOBBER_EXISTING_OUTPUTS = False

PROCESSING_ROOT = config.config.get('subject_data', 'processing_root')

ATLAS_BASE = config.config.get('subject_data', 'atlas_base')
flair_atlas = config.config.get('subject_data', 'flair_atlas')

ncerebro_base = os.path.dirname(os.path.dirname(config.config.get('Binaries', 'nCEREBRO')))
cerebro_atlas = os.path.join(ncerebro_base, "fixtures","iso_flair_template_intres_brain.nii.gz")
cerebro_atlas_affine = os.path.join(ncerebro_base, "fixtures","new_to_old0GenericAffine.mat")


def check_fluid_attenuation(input_filename, seg_filename, output_filename):
    import nibabel as nib
    import numpy as np
    data = nib.load(input_filename).get_data()
    seg = nib.load(seg_filename).get_data()
    ventricle = np.median(data[np.logical_and(seg==4, seg==43)])
    wm = np.median(data[np.logical_and(seg==2, seg==41)])
    with open(output_filename, 'w') as f:
        if ventricle >= wm:
            f.write('1')
        else:
            f.write('0')
if __name__ == '__main__':

    ########################
    ### Argument parsing ###
    ########################
    USAGE = '%s <subj> <smoothness regularization> <field regularization> <out folder> [<subj list>]' % sys.argv[0]

    if len(sys.argv) not in [5,6]:
        print(USAGE)
        sys.exit(1)

    subj = sys.argv[1]
    mod = 'flair'
    # Regularization parameters for ANTS
    regularization = float(sys.argv[2])
    regularization2 = float(sys.argv[3])

    # where the data lives
    data_subfolder = sys.argv[4]

    #############################
    ### Set up atlas and data ###
    #############################

    BASE = os.path.join(PROCESSING_ROOT, data_subfolder)
    ## Atlas
    atlas = pb.Dataset(ATLAS_BASE, flair_atlas, None)

    ## Subject data
    dataset = pb.Dataset(
                BASE,
                # How are the inputs to the pipeline stored?
                os.path.join(BASE , '{subj}/original/{modality}_1/{subj}_{modality}_{feature}'),
                # How should intermediate files be stored?
                #os.path.join(BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}'),
                os.path.join(BASE, '{subj}/images/{subj}_{modality}_{feature}{modifiers}'),
                log_template=os.path.join(BASE, '{subj}/logs/'),
                pickle_template=os.path.join(BASE, '{subj}/images/{subj}_{modality}_zoom_log.pickle')
                )

    #dataset.add_mandatory_input(modality='t1', feature='raw')
    #dataset.add_mandatory_input(modality=mod, feature='img')
    dataset.add_mandatory_input(modality=mod, feature='raw')
    #dataset.get_original(subj=subj, modality='t1', feature='raw')

    #############################
    ### Registration pipeline ###
    #############################

    ###
    flair_input = dataset.get_original(subj=subj, modality=mod, feature='raw')
    atlas_img = os.path.join(ATLAS_BASE, flair_atlas)

    modifiers = ''
    neuronbe = pb.custom.NeuronBECommand(
            "Brain extraction with NeuronBE",
            input=flair_input,
            output=dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers+'_brain'),
            out_mask=dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers+'_brainmask_01'), 
            out_intres=dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers + '_brain_matchwm'),
            out_gmwm_mask=dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers + '_gmwm_mask'),
            out_refined_mask=dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers+'_brainmask_02'))
    modifiers += '_brain_matchwm'
    
    upsample = pb.NiiToolsUpsampleCommand(
                 "Upsample flair image",
                 input=dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers),
                 output=dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers+'_upsample'),
                 zoom_values_file=dataset.get_pickle_file(subj=subj, modality=mod)
                 )
    modifiers += '_upsample'

    subj_final_img = dataset.get(subj=subj, modality=mod, feature='img', modifiers=modifiers)
    # downsampled_final_image = pb.NiiToolsDownsampleCommand(
    #          "Downsample final flair image",
    #          input=subj_final_img,
    #          output=dataset.get(subj=subj, modality=mod, feature='pipeline', modifiers=''),
    #          zoom_values_file=dataset.get_pickle_file(subj=subj, modality=mod)
    #          )

    ###### Final atlas -> subject registration
    reg_file_prefix = subj + '_' + mod + '_'

    forward_reg = pb.ANTSCommand(
            "Register label-blurred flair atlas to subject",
            moving=atlas_img,
            fixed=subj_final_img,
            output_folder=os.path.join(dataset.get_folder(subj=subj), 'reg'),
            output_prefix=os.path.join(dataset.get_folder(subj=subj), 'reg', reg_file_prefix),
            method='affine',
            )

    pb.ANTSWarpCommand.make_from_registration(
            "Warp subject image to atlas space using warp",
            moving=subj_final_img,
            reference=atlas_img,
            output_filename=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_atlas'),
            registration=forward_reg,
            inversion='inverse',
            )

    ############################
    # Segmentation
    ############################

    # initialize registration from atlas to the one cerebro was trained on
    caa_reg = copy.copy(forward_reg)
    caa_reg.forward_warp_string = cerebro_atlas_affine
    caa_reg.backward_warp_string = '-i ' + cerebro_atlas_affine
    tmp = list(caa_reg.inputs)
    tmp[1] = cerebro_atlas
    tmp[2] = os.path.join(ATLAS_BASE,flair_atlas)
    caa_reg.inputs = set(tmp)
    caa_reg.outfiles = [cerebro_atlas_affine]
    caa_reg.output_prefix = cerebro_atlas_affine.split('Affine')[0]

    pb.ANTSWarpCommand.make_from_registration(
            "Warp subject in atlas image to original atlas space in which nCerebro was trained",
            moving=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_atlas'),
            reference=cerebro_atlas,
            output_filename=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_orig_atlas'),
            registration=caa_reg,
            )

    cerebro = pb.custom.nCerebroCommand(
            "Segment WMH with cerebro",
            input=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_orig_atlas'),
            output=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_orig_atlas'))

    pb.ANTSWarpCommand.make_from_registration(
            "Warp subject in orig atlas image to atlas space using warp",
            moving=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_orig_atlas'),
            reference=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_atlas'),
            output_filename=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_atlas'),
            registration=caa_reg,
            inversion='inverse',
            )

    pb.ANTSWarpCommand.make_from_registration(
            "Warp leuk seg to upsampled img space using warp",
            moving=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_atlas'),
            reference=subj_final_img,
            output_filename=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_upsampled_subject'),
            registration=forward_reg,
            )

    pb.NiiToolsDownsampleCommand(
             "Downsample flair image",
             input=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_upsampled_subject'),
             output=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_subject'),
             zoom_values_file=dataset.get_pickle_file(subj=subj, modality=mod)
             )

    pb.NiiToolsBinarize(
    			"Binarize image",
    			input=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_in_subject'),
                output=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_leuk_seg_bin'),
                threshold=0.5
    			)


    for path in [os.path.join(BASE,subj,'images'),
            os.path.join(BASE,subj,'images','reg'),
            dataset.get_log_folder(subj=subj)]:
        try:
            os.mkdir(path)
        except:
            pass

    ### Generate script file and SGE qsub file
    tracker = tracking.Tracker(pb.Command.all_commands, pb.Dataset.all_datasets)
    tracker.compute_dependencies()

    ###
    log_folder = dataset.get_log_folder(subj=subj)
    pb.Command.generate_code_from_datasets([dataset, atlas], log_folder, subj, sge=False,
            wait_time=0, tracker=tracker)

