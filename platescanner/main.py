from platescanner.modes import *

def main():
    import sys
    args = sys.argv
    # remove path to file from args
    args.pop(0)

    # match mode
    if len(args) == 0:
        print("Error: Please, specify the mode")
        exit(-1)

    match args[0]:
        case 'train':
            mode_train(args)
        case 'predict':
            mode_predict(args)
        case 'validate':
            mode_validate(args)
        case 'find_confidence':
            mode_find_confidence(args)
        case 'make_calibration_dataset':
            mode_make_calibration_dataset(args)
        case 'help' | '-h' | 'h':
            print("""
train:
    '-weights_path'     : None                      # path to model's weights file 
    '-dataset_path'     : None                      # path to dataset directory  
    '-use_clearml'      : False                     # use clearml for training

predict:
    '-weights_path'     : None                      # path to model's weights file
    '-dataset_path'     : None                      # path to dataset directory
    '-output_path'      : None                      # path to output directory (should exists and be empty)
    '-confidence_level' : DEFAULT_CONFIDENCE_LEVEL  # confidence level
    '-detection_only'   : False                     # True - detection only; False - detection + recognition

validate:
    '-weights_path'     : None                      # path to model's weights file
    '-dataset_path'     : None                      # path to dataset directory
    '-output_path'      : None                      # path to output directory (should exists and be empty)
    '-confidence_level' : DEFAULT_CONFIDENCE_LEVEL  # confidence level
    '-detection_only'   : None                      # None: detection + recognition; True - detection only; False - recognition

find_confidence:
    '-weights_path'     : None                      # path to model's weights file
    '-dataset_path'     : None                      # path to dataset directory
    '-iters'            : 7                         # number of iterations
    '-target_precision' : 0.95                      # target precision

make_calibration_dataset:
    '-dataset_path'     : None                      # path to dataset directory

""")
        case _:
            print("Please, choose any valid mode")


if __name__ == '__main__':
    main()
