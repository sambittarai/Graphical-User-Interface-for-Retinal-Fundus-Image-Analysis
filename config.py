import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # in/out
    # parser.add_argument('--outf', default='/content/drive/MyDrive/Retinal_Vessel_Segmentation/Experiments',
    #                     help='trained model will be saved at here') #Not use
    # parser.add_argument('--save', default='UNet_vessel_seg',
    #                     help='save name of experiment in args.outf directory') #Not use
    # Retinal Vessel Segmentation
    parser.add_argument('--Vessel_best_model_path', default='G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/GUI/Network_Weights/Retinal_Vessel_Segmentation/best_model.pth',
                        help='directory of best model path for Retinal Vessel Segmentation')
    # Optic Disc Segmentation
    parser.add_argument('--OD_best_model_path', default='G:/IIT_MADRAS_DD/Semesters/10th_sem/DDP_new_topic/My work/GUI/Network_Weights/Optic_Disc_Segmentation/best_model.pth',
                        help='directory of best model path for OD Segmentation')

    # model parameters
    parser.add_argument('--in_channels', default=1,type=int,
                        help='input channels of model')
    parser.add_argument('--classes', default=2,type=int, 
                        help='output channels of model')

    parser.add_argument('--batch_size_VS', default=64,
                       type=int, help='batch size for Vessel Segmentation')
    parser.add_argument('--batch_size_OD', default=32,
                       type=int, help='batch size for Optic Disc Segmentation')

    # inference
    parser.add_argument('--test_patch_height_VS', default=64)
    parser.add_argument('--test_patch_width_VS', default=64)
    parser.add_argument('--stride_height_VS', default=32)
    parser.add_argument('--stride_width_VS', default=32)

    parser.add_argument('--test_patch_height_OD', default=128)
    parser.add_argument('--test_patch_width_OD', default=128)
    parser.add_argument('--stride_height_OD', default=64)
    parser.add_argument('--stride_width_OD', default=64)

    # hardware setting
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use GPU calculating')

    args = parser.parse_args()

    return args