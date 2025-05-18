import pprint

from cvtk.bbox import Bbox_CWH
from cvtk.bbox.adapters import AdapterCwh2xy

from platescanner import DATASETS_PATH
from platescanner.pipelines import PTPipeline
from platescanner.utils import to_mvp
from platescanner.utils.recognition_metrics import evaluate_metrics


def main():
    adapter = AdapterCwh2xy()

    dataset = to_mvp(DATASETS_PATH / 'converted' / 'test1')
    pipeline = PTPipeline()
    result = pipeline.predict(dataset)

    # convert to eval metrics
    predict_result = {
        image_name: [(bbox, bbox.value) for bbox in result[i]]
        for i, image_name in enumerate(dataset.images['test'])
    }
    ground_truth_result = {
        image_name: [
            (
                adapter.compute(
                    Bbox_CWH(bbox_desc['points'], value=bbox_desc['recognition_text'])
                ),
                bbox_desc['recognition_text']
            )
            for bbox_desc in dataset.attributes['test'][image_name]["Detection"]["bboxes"]]
        for i, image_name in enumerate(dataset.images['test'])
    }

    metrics = evaluate_metrics(ground_truth_result, predict_result)
    pprint.pp(metrics)


if __name__ == '__main__':
    main()
