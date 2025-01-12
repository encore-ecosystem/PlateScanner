from platescanner.modes import *


def main(args):
    # remove path to file from args
    args.pop(0)

    # match mode
    assert len(args) != 0, "Please, specify the mode"
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
        case _:
            print("Please, choose any valid mode")


if __name__ == '__main__':
    import sys
    main(sys.argv)
