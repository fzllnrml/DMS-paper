"""Cached fixed reference evaluation and validation-calibrated memorization audit."""
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
from .common import fingerprint, save_pt
from .evaluation import SupportEvaluator, nearest_neighbors, support_percentiles


def memorization_metrics(train_distance, test_distance, validation_train, validation_test, quantile=.01):
    train = torch.as_tensor(train_distance).float()
    test = torch.as_tensor(test_distance).float()
    vt = torch.as_tensor(validation_train).float()
    vs = torch.as_tensor(validation_test).float()
    if train.shape != test.shape or vt.shape != vs.shape or not train.numel() or not vt.numel():
        raise ValueError('Invalid memorization distances')
    if not 0 < quantile < 1:
        raise ValueError('Invalid close-reference quantile')
    threshold = torch.quantile(vt, quantile)
    # Raw train/test distances depend on reference cardinalities. Compare with
    # validation images evaluated against these exact same reference sets.
    gap = train - test
    reference_gap = vt - vs
    return dict(nearest_train_distance=train, nearest_test_distance=test,
                train_minus_test_distance=gap,
                gap_percentiles=support_percentiles(gap, reference_gap)['distance_percentile'],
                unusually_close=train <= threshold,
                close_threshold=float(threshold), close_reference_quantile=quantile,
                reference_validation_train=vt, reference_validation_test=vs,
                summary=dict(mean_nearest_train_distance=float(train.mean()),
                             mean_nearest_test_distance=float(test.mean()),
                             mean_train_minus_test_distance=float(gap.mean()),
                             validation_mean_train_minus_test_distance=float(reference_gap.mean()),
                             fraction_unusually_close_train=float((train <= threshold).float().mean()),
                             train_test_gap_percentile=float(np.mean(support_percentiles(gap, reference_gap)['distance_percentile']))),
                interpretation='Validation-calibrated proximity diagnostic, not proof of copying or its absence.')


class PublicationSupport(SupportEvaluator):
    def __init__(self, data, evaluator, config, evaluator_sha):
        self.data = data
        self.evaluator = evaluator
        self.k = config['knn_k']
        self.train = data.images('train', config['train_limit'])
        self.validation = data.images('validation', config['validation_limit'])
        self.test = data.images('test') if config['include_test'] else None
        hashes = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()
                  for k, v in [('train', self.train), ('validation', self.validation), ('test', self.test)]
                  if v is not None}
        key = fingerprint(dict(version=1, pixels=hashes, evaluator=evaluator_sha, k=self.k,
                               torch=str(torch.__version__), evaluator_device=str(evaluator.device) if evaluator else None))
        cache = Path(config.get('reference_cache', 'runs/mnist_experiments/publication_reference')) / (key + '.pt')
        if cache.exists():
            state = torch.load(cache, map_location='cpu', weights_only=True)
            if state['key'] != key:
                raise ValueError('Reference cache mismatch')
            self.train_features = state['train_features']
            self.reference = state['reference']
            self.validation_test = state['validation_test']
        else:
            self.train_features = evaluator.features(self.train) if evaluator else None
            validation_features = evaluator.features(self.validation) if evaluator else None
            self.reference = {}
            for name, q, r in [('pixel', self.validation, self.train), ('feature', validation_features, self.train_features)]:
                if q is None:
                    continue
                distances, _ = nearest_neighbors(q, r, self.k)
                if name == 'pixel':
                    distances /= math.sqrt(784)
                self.reference[name] = dict(nearest=distances[:, 0], knn=distances.mean(1))
            self.validation_test = (nearest_neighbors(self.validation, self.test, 1)[0][:, 0] / math.sqrt(784)
                                    if self.test is not None else None)
            save_pt(cache, dict(key=key, train_features=self.train_features, reference=self.reference,
                               validation_test=self.validation_test))
        self.metadata = dict(train_reference_count=len(self.train), validation_reference_count=len(self.validation),
                             test_reference_count=len(self.test) if self.test is not None else 0,
                             train_is_subset=config['train_limit'] is not None,
                             validation_is_subset=config['validation_limit'] is not None,
                             k=self.k, pixel_distance_units='RMS normalized pixels', feature_distance_units='L2 frozen features',
                             test_memorization_audit=self.test is not None, test_used_for_tuning=False,
                             cache_key=key, unusually_close_quantile=config.get('unusually_close_quantile', .01))

    def evaluate(self, images, initial_count):
        result = super().evaluate(images, initial_count)
        if self.test is not None:
            result['memorization'] = memorization_metrics(result['pixel']['nearest_distance'],
                result['test_nearest']['distance'], self.reference['pixel']['nearest'], self.validation_test,
                self.metadata['unusually_close_quantile'])
        return result
