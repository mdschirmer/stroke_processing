#!/usr/bin/env python
import os
import sys

import pipebuilder as pb
from pipebuilder import tracking
import pipebuilder.custom

from stroke_processing.tools import config
cwd = os.path.dirname(os.path.abspath(__file__))

ATLAS_MODALITY = 't1'
DWI_INTENSITY = '210'

WMH_THRESHOLD = 430
STROKE_THRESHOLD = 1
STROKE_THRESHOLD = 1

CLOBBER_EXISTING_OUTPUTS = False

PROCESSING_ROOT = config.config.get('subject_data', 'processing_root')

ATLAS_BASE = config.config.get('subject_data', 'atlas_base')
flair_atlas = config.config.get('subject_data', 'flair_atlas')


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
    print(BASE)
    ## Atlas

    atlas = pb.Dataset(ATLAS_BASE, flair_atlas, None)
    buckner = pb.Dataset(ATLAS_BASE, 'buckner61{feature}{extension}', None)

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
    neuronss = pb.custom.NeuronSSCommand(
            "Brain extraction with NeuronSS",
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
            output_filename=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_atlas_affine'),
            registration=forward_reg,
            inversion='inverse',
            )
    affine_img = dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_atlas_affine')

#    forward_reg = pb.ANTSCommand(
#            "Register label-blurred flair atlas to subject",
#            moving=atlas_img,
#            fixed=affine_img,
#            output_folder=os.path.join(dataset.get_folder(subj=subj), 'reg'),
#            output_prefix=os.path.join(dataset.get_folder(subj=subj), 'reg', reg_file_prefix),
#            transformation='Diff',
#            histogrammatching=0,
#            metric='CC',
#            radiusBins=4,
#            regularization='Gauss[%0.3f,%0.3f]' % (regularization,regularization2),
#            method='nonlinear',
#            nonlinear_iterations="401x401x401",
#            )

#    pb.ANTSWarpCommand.make_from_registration(
#            "Warp subject image to atlas space using warp",
#            moving=affine_img,
#            reference=atlas_img,
#            output_filename=dataset.get(subj=subj, modality=mod, feature='img', modifiers='_in_atlas_nonlinear'),
#            registration=forward_reg,
#            inversion='inverse',
#            )

    # label_warp = pb.ANTSWarpCommand.make_from_registration(
    #         "Warp atlas labels to subject space using  warp",
    #         moving=buckner.get_original(feature='_seg'),
    #         reference=subj_final_img,
    #         registration=forward_reg,
    #         output_filename=dataset.get(subj=subj, modality=mod, feature='atlas_labels', modifiers='_in_subject_seg'),
    #         useNN=True,
    #         )

    # downsampled_atlas_label_in_sub = pb.NiiToolsDownsampleCommand(
    #         "Downsample atlas labels in subject space",
    #         input=dataset.get(subj=subj, modality=mod, feature='atlas_labels', modifiers='_in_subject_seg'),
    #         output=dataset.get(subj=subj, modality=mod, feature='pipeline', modifiers='_atlas_labels'),
    #         zoom_values_file=dataset.get_pickle_file(subj=subj, modality=mod)
    #         )

    # pb.ANTSWarpCommand.make_from_registration(
    #         "Warp atlas image to subject space using  warp",
    #         moving=atlas_img,
    #         reference=subj_final_img,
    #         output_filename=dataset.get(subj=subj, modality=mod, feature='atlas_img', modifiers='_in_subject'),
    #         registration=forward_reg)


    # filename = os.path.basename(label_warp.outfiles[0]).split('.')[0]

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

