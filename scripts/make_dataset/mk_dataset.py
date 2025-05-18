# 1. Load pipeline
# 2. For each image in the dataset
# 3.    apply pipeline to image
# 4.    save image into buffer
#    end for
# 5. split buffer into train / test / eval sets
# 6. convert to mvp format


from nodeflow.converter import Optional
import numpy as np
from cvtk.interfaces import Bbox
from cvtk.bbox.adapters import Adapter2xyCwh, AdapterCwh2xy
from cvtk.bbox.bboxes import Bbox_CWH
from cvtk.supported_datasets.mvp.dataset import MVP_Dataset
from prologger import ConsoleLogger

from platescanner import DATASETS_PATH, EXPERIMENTATOR_PATH
from platescanner.pipelines import PTPipeline

from nodeflow import Converter

import shutil


IMAGES_PATH = DATASETS_PATH / "raw_images" / "test1"
SAVE_TO     = DATASETS_PATH / "converted"  / "test1"

def describe_bbox(bbox: Bbox):
    converter = Converter(
        adapters = [Adapter2xyCwh(), AdapterCwh2xy()]
    )
    conv_bbox: Optional[Bbox_CWH] = converter.convert(variable=bbox, to_type=Bbox_CWH)
    if conv_bbox is None:
        raise ValueError("Failed to convert bbox")
    return {
        "superclasses": [],
        "bbox_type": "bb",
        "class_name": "plate",
        "points": conv_bbox.bbox,
        "recognition_text": conv_bbox.value
    }

def main():
    ttv = [0.0, 1.0, 0.0]

    logger = ConsoleLogger()
    # Use this if pipeline not exists:
    # PTPipeline().save(EXPERIMENTATOR_PATH / "real_fit_ptp")

    # step 1
    logger.info("Step 1")
    pipeline = PTPipeline.load(src_path=EXPERIMENTATOR_PATH / "real_fit_ptp")

    # step 2
    logger.info("Step 2")
    processed = {}
    for image_path in IMAGES_PATH.glob("*"):
        if not image_path.is_file():
            continue

        logger.info(f"Step 3 and 4 for {image_path.name}")
        pairs = pipeline.predict_on_image(image_path)
        processed[image_path] = pairs

    # step 5
    logger.info("Step 5")
    ttv = np.abs(np.array(ttv))
    ttv = ttv / np.linalg.norm(ttv)

    image_names = list(processed.keys())
    n = len(image_names)
    train = image_names[:int(n * ttv[0])]
    test  = image_names[int(n * ttv[0]):int(n * ttv[1])]
    valid = image_names[int(n * ttv[1]):]

    # step 6
    logger.info("Step 6")
    dataset = MVP_Dataset(
        path = SAVE_TO,
        manifest = {
            "Info": {
                "sources": [ "Converted from raw images", ],
                "version": "1.0",
            }
        },
        images = {
            "train" : {image_path.stem: image_path for image_path in train},
            "test"  : {image_path.stem: image_path for image_path in test},
            "valid" : {image_path.stem: image_path for image_path in valid},
        },
        attributes = {
            "train": {
                f"{image_path.stem}.txt": {
                    "Detection": {
                        "bboxes": [
                            describe_bbox(bbox) for bbox in processed[image_path]
                        ]
                    },
                    "Segmentation": {
                        "polygons": []
                    }
                }
                for image_path in train
            },
            "test": {
                f"{image_path.stem}.txt": {
                    "Detection": {
                        "bboxes": [
                            describe_bbox(bbox) for bbox in processed[image_path]
                        ]
                    },
                    "Segmentation": {
                        "polygons": []
                    }
                }
                for image_path in test
            },
            "valid": {
                f"{image_path.stem}.txt": {
                    "Detection": {
                        "bboxes": [
                            describe_bbox(bbox) for bbox in processed[image_path]
                        ]
                    },
                    "Segmentation": {
                        "polygons": []
                    }
                }
                for image_path in valid
            },
        }
    )

    shutil.rmtree(SAVE_TO)
    dataset.write(SAVE_TO)

    logger.info("Complete!")


if __name__ == "__main__":
    main()
