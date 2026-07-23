"""
Streamlit Dashboard Integration Tests.
"""

import pytest
import app


def test_dashboard_import():
    assert hasattr(app, "main")


def test_demo_image_loader():
    demo_img = app.load_demo_document()
    assert len(demo_img.shape) == 3 and demo_img.shape[0] > 0 and demo_img.shape[1] > 0
