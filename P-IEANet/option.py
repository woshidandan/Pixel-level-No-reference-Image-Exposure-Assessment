import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument('--path_to_ava_txt', type=str, default="",
                        help='directory to csv_folder')

    parser.add_argument('--path_to_images', type=str, default='F:/IEA/IEA_img',
                        help='directory to images')

    parser.add_argument('--path_to_save_csv', type=str,default="",
                        help='directory to csv_folder')

    parser.add_argument('--experiment_dir_name', type=str, default='.',
                        help='directory to project')

    parser.add_argument('--path_to_model_weight', type=str, default='',
                        help='directory to pretrain model')

    parser.add_argument('--init_lr', type=int, default=0.00003, help='learning_rate'
                        )
    parser.add_argument('--num_epoch', type=int, default=100, help='epoch num for train'
                        )
    parser.add_argument('--batch_size', type=int,default=16,help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=0, help ='num_workers',
                        )
    parser.add_argument('--gpu_id', type=str, default='0', help='which gpu to use')


    args = parser.parse_args()
    return args