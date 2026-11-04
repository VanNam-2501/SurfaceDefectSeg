"""Test training mechanisms independently of any claimed accuracy improvement."""
from pathlib import Path
import math
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inspection_research.feature_style import (FeatureStyleMix, StyleResNet18,
    style_partners, jensen_shannon_from_logits, paired_training_loss)


class FeatureStyleTests(unittest.TestCase):
    def test_same_class_pairing_never_uses_other_label_or_self_when_possible(self):
        labels = torch.tensor([0, 0, 1, 1, 2])
        partners = style_partners(labels, 5, "same_class", device="cpu")
        self.assertTrue(torch.equal(labels[partners], labels))
        self.assertTrue(torch.equal(partners, torch.tensor([1, 0, 3, 2, 4])))

    def test_cross_domain_pairing_requires_same_class_and_different_domain(self):
        labels, domains = torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0, 1, 2])
        partners = style_partners(labels, 4, "cross_domain_same_class", domains, "cpu")
        for index, partner in enumerate(partners):
            if index != partner:
                self.assertEqual(labels[index], labels[partner])
                self.assertNotEqual(domains[index], domains[partner])
        self.assertEqual(int(partners[-1]), 3)

    def test_missing_domain_metadata_is_not_silently_invented(self):
        with self.assertRaisesRegex(ValueError, "observed"):
            style_partners(torch.tensor([0, 0]), 2, "cross_domain_same_class", device="cpu")

    def test_exact_forward_content_identity_under_known_style(self):
        torch.manual_seed(3)
        features = torch.randn(2, 3, 5, 5, dtype=torch.float64)
        features[1] = features[1] * 3 + 5
        module = FeatureStyleMix(strength=.7, probability=1).train()
        partners, coefficients = torch.tensor([1, 0]), torch.tensor([.2, .4], dtype=torch.float64)
        result = module(features, partners, coefficients, force=True)
        mean = features.mean((2, 3), keepdim=True)
        std = (features.var((2, 3), unbiased=False, keepdim=True) + module.eps).sqrt()
        weight = .7 * (1 - coefficients[:, None, None, None])
        mixed_mean = mean + weight * (mean[partners] - mean)
        mixed_std = std + weight * (std[partners] - std)
        self.assertTrue(torch.allclose((result - mixed_mean) / mixed_std, (features - mean) / std, atol=1e-12))

    def test_zero_strength_and_eval_are_exact_identities(self):
        features = torch.randn(3, 4, 8, 8)
        self.assertTrue(torch.equal(FeatureStyleMix(strength=0).train()(features), features))
        self.assertTrue(torch.equal(FeatureStyleMix().eval()(features), features))

    def test_no_valid_partner_is_exact_identity(self):
        features = torch.randn(3, 4, 8, 8)
        output = FeatureStyleMix(probability=1).train()(features, torch.arange(3), force=True)
        self.assertTrue(torch.equal(output, features))

    def test_donor_statistics_do_not_receive_recipient_gradient(self):
        features = torch.randn(2, 3, 4, 4, requires_grad=True)
        output = FeatureStyleMix(probability=1).train()(features, torch.tensor([1, 0]), torch.tensor([.3, .3]), force=True)
        output[0].sum().backward()
        self.assertGreater(float(features.grad[0].abs().sum()), 0)
        self.assertEqual(float(features.grad[1].abs().sum()), 0)

    def test_constant_features_are_finite(self):
        output = FeatureStyleMix(probability=1).train()(torch.ones(2, 3, 4, 4), torch.tensor([1, 0]), force=True)
        self.assertTrue(torch.isfinite(output).all())

    def test_js_is_symmetric_bounded_and_zero_on_identical_predictions(self):
        first, second = torch.tensor([[1000., -1000.]]), torch.tensor([[-1000., 1000.]])
        js = jensen_shannon_from_logits(first, second)
        self.assertAlmostEqual(float(js), math.log(2), places=6)
        self.assertTrue(torch.equal(js, jensen_shannon_from_logits(second, first)))
        self.assertAlmostEqual(float(jensen_shannon_from_logits(first, first)), 0, places=7)

    def test_paired_erm_baseline_reduces_to_ce_for_identical_logits(self):
        logits, labels = torch.randn(4, 5), torch.tensor([0, 1, 2, 3])
        actual, _ = paired_training_loss(logits, logits, labels, consistency_weight=0)
        self.assertTrue(torch.allclose(actual, F.cross_entropy(logits, labels)))

    def test_js_local_quadratic_expansion(self):
        # Test the probability-space expansion JS(p,p+e v) = e^2/8 sum v^2/p + O(e^3).
        p = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
        direction = torch.tensor([[.1, -.2, .1]], dtype=torch.float64)
        eps = 1e-4
        q = p + eps * direction
        actual = jensen_shannon_from_logits(p.log(), q.log())
        leading = eps**2 / 8 * (direction.square() / p).sum()
        self.assertAlmostEqual(float(actual / leading), 1, places=4)

    def test_collapsed_predictions_are_penalized_by_supervision(self):
        labels = torch.tensor([0, 1, 2])
        collapsed = torch.zeros(3, 3)
        correct = torch.eye(3) * 20
        bad, _ = paired_training_loss(collapsed, collapsed, labels)
        good, _ = paired_training_loss(correct, correct, labels)
        self.assertGreater(float(bad), float(good) + 1)

    def test_model_eval_equals_plain_resnet_without_labels(self):
        torch.manual_seed(4)
        model = StyleResNet18(3, probability=1).eval()
        images = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            expected = model.network(images)
            actual = model(images, augment=True, pairing="same_class")
        self.assertTrue(torch.equal(actual, expected))

    def test_real_resnet_backward_supports_product_specific_class_counts(self):
        for count in [2, 5, 6]:
            torch.manual_seed(4)
            model = StyleResNet18(count, probability=1).train()
            images, labels = torch.randn(4, 3, 32, 32), torch.tensor([0, 0, 1, 1])
            clean = model(images)
            mixed = model(images, labels=labels, augment=True, pairing="same_class")
            loss, metrics = paired_training_loss(clean, mixed, labels, .2)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.network.conv1.weight.grad.abs().sum()), 0)
            self.assertTrue(all(torch.isfinite(value) for value in metrics.values()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
