"""
PyTorch Document Type Classifier Model.

Classifies incoming document images into structural document categories
(e.g., Invoice, Passport, ID Card, Utility Bill, Tax Form) to route downstream rules.
"""

from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torchvision.models as models

from src.utils.logger import get_logger

logger = get_logger("DocumentClassifier")


class DocumentClassifier(nn.Module):
    """
    ResNet-18 transfer learning classifier for document categorization.
    """

    DEFAULT_CATEGORIES = ["Invoice", "Passport", "ID Card", "Utility Bill", "Tax Form", "Generic Document"]

    def __init__(self, categories: Optional[List[str]] = None, pretrained: bool = False) -> None:
        super().__init__()
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.num_classes = len(self.categories)

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # Omit FC layer
        self.fc = nn.Linear(512, self.num_classes)

        logger.info(f"Initialized DocumentClassifier ({self.num_classes} document categories)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Pass.

        Args:
            x (torch.Tensor): Tensor [Batch, 3, Height, Width].

        Returns:
            torch.Tensor: Category logits [Batch, Num_Classes].
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x).squeeze(-1).squeeze(-1) # [B, 512]
        return self.fc(features)

    def predict_category(self, image_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Predicts top category and confidence probability for single image tensor.
        """
        self.eval()
        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            logits = self.forward(image_tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)

            category = self.categories[top_idx.item()]
            return category, float(top_prob.item())


def classify_document_multimodal(image: Optional[torch.Tensor] = None, image_np: Optional[Any] = None, ocr_text: str = "") -> Tuple[str, Dict[str, float]]:
    """
    Multi-modal Visual + Text Feature Classifier with Out-of-Domain (OOD) Rejection.

    Detects valid document types (Passport, PAN, Aadhaar, Driving License, Invoice, Receipt, Cheque)
    and rejects non-document images (animals, photos, landscapes, objects).
    """
    import cv2
    import re
    import numpy as np

    categories = [
        "Passport",
        "PAN Card",
        "Aadhaar Card",
        "Driving License",
        "Invoice",
        "Receipt",
        "Cheque",
        "Invalid / Non-Document Image"
    ]

    text_upper = ocr_text.upper().strip() if ocr_text else ""

    # Convert Image
    img_bgr = None
    if image_np is not None:
        img_bgr = image_np
    elif image is not None and isinstance(image, torch.Tensor):
        try:
            arr = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_bgr = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:
            img_bgr = None

    # 1. OpenCV Document Layout & Text Contour Analysis
    has_doc_layout = False
    if img_bgr is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSS_C, cv2.THRESH_BINARY_INV, 15, 8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            text_boxes = [c for c in contours if cv2.boundingRect(c)[2] > 15 and cv2.boundingRect(c)[3] > 6]
            has_doc_layout = len(text_boxes) >= 3
        except Exception:
            has_doc_layout = False

    # 2. Strict Human Face Detection with Skin-Tone Validation
    has_human_face = False
    if img_bgr is not None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))
            
            if len(faces) > 0:
                hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
                for (x, y, w, h) in faces:
                    face_hsv = hsv[y:y+h, x:x+w]
                    skin_mask = cv2.inRange(face_hsv, np.array([0, 30, 60]), np.array([25, 200, 255]))
                    skin_ratio = np.count_nonzero(skin_mask) / (w * h)
                    if skin_ratio > 0.15:
                        has_human_face = True
                        break
        except Exception:
            has_human_face = False

    # Check for text keywords
    passport_kw = ["PASSPORT", "PASSEPORT", "REPUBLIC", "NATIONALITY", "SURNAME", "GIVEN NAMES", "DATE OF BIRTH", "DATE OF EXPIRY", "AUTHORITY", "EOLIE", "P<", "COUNTRY"]
    passport_matches = sum(1 for kw in passport_kw if kw in text_upper)
    has_mrz = ("<<" in text_upper) or ("P<" in text_upper) or bool(re.search(r"P<[A-Z0-9<]{5,}", text_upper))

    pan_kw = ["PERMANENT ACCOUNT NUMBER", "INCOME TAX", "GOVT. OF INDIA", "GOVT OF INDIA", "FATHER'S NAME", "DEPARTMENT", "PAN", "आयकर विभाग", "INCOME", "TAX", "PERMANENT", "ACCOUNT", "CARD", "SIGNATURE"]
    pan_matches = sum(1 for kw in pan_kw if kw in text_upper)

    aadhaar_kw = ["AADHAAR", "UIDAI", "UNIQUE IDENTIFICATION", "ENROLMENT", "HELP@UIDAI", "GOVERNMENT OF INDIA", "MALE", "FEMALE", "YEAR OF BIRTH", "YOB", "FATHER", "ADDRESS"]
    aadhaar_matches = sum(1 for kw in aadhaar_kw if kw in text_upper)

    dl_kw = ["DRIVING", "LICENCE", "LICENSE", "DL NO", "AUTHORIZATION", "TRANSPORT", "MOTOR", "VEHICLES", "UNION OF INDIA", "DL"]
    dl_matches = sum(1 for kw in dl_kw if kw in text_upper)

    invoice_kw = [
        "INVOICE", "TAX INVOICE", "BILL TO", "SHIP TO", "SUBTOTAL", "AMOUNT DUE", "GSTIN", "VAT", "DUE DATE",
        "ORDER", "SUMMARY", "ORDER SUMMARY", "ORDER SUBTOTAL", "GRAND TOTAL", "SOLD BY", "PAYMENT", "PAYMENT METHOD",
        "BHIM", "UPI", "INR", "RS", "ITEM", "ARRIVING", "DELIVERY", "ITEMS", "DISCOUNT", "TAX", "TOTAL", "PRICING", "UNIT PRICE", "BILLING"
    ]
    invoice_matches = sum(1 for kw in invoice_kw if kw in text_upper)

    receipt_kw = ["RECEIPT", "TOTAL", "CASH", "CHANGE", "STORE", "POS", "THANK YOU", "MERCHANT"]
    receipt_matches = sum(1 for kw in receipt_kw if kw in text_upper)

    cheque_kw = ["PAY", "RUPEES", "ACCOUNT", "IFS", "CHEQUE", "BANK", "DRAWER", "PAYEE"]
    cheque_matches = sum(1 for kw in cheque_kw if kw in text_upper)

    total_matches = (passport_matches + pan_matches + aadhaar_matches + dl_matches + invoice_matches + receipt_matches + cheque_matches)

    # REJECTION CRITERIA: Out-Of-Domain Rejection Filter
    # Only reject if NO text layout contours exist AND NO document keywords match AND NO human face/MRZ exists
    if not has_doc_layout and total_matches == 0 and not has_human_face and not has_mrz:
        probs = {c: 0.0 for c in categories}
        probs["Invalid / Non-Document Image"] = 1.0
        return "Invalid / Non-Document Image", probs

    # Scoring valid categories
    scores = {c: 1.0 for c in categories if c != "Invalid / Non-Document Image"}

    if has_human_face:
        scores["Passport"] += 10.0
        scores["PAN Card"] += 8.0
        scores["Aadhaar Card"] += 8.0
        scores["Driving License"] += 8.0

    if has_mrz or passport_matches >= 1:
        scores["Passport"] += 40.0 + (passport_matches * 10.0)

    if pan_matches >= 1:
        scores["PAN Card"] += 35.0 + (pan_matches * 8.0)

    if aadhaar_matches >= 1:
        scores["Aadhaar Card"] += 35.0 + (aadhaar_matches * 8.0)

    if dl_matches >= 1:
        scores["Driving License"] += 35.0 + (dl_matches * 8.0)

    if invoice_matches >= 1 or (has_doc_layout and not has_human_face):
        scores["Invoice"] += 25.0 + (invoice_matches * 8.0)

    if receipt_matches >= 1 and not has_human_face:
        scores["Receipt"] += 25.0 + (receipt_matches * 8.0)

    if cheque_matches >= 2 and not has_human_face:
        scores["Cheque"] += 25.0 + (cheque_matches * 8.0)

    # Normalize
    max_val = max(scores.values())
    exp_scores = {k: float(np.exp(v - max_val)) for k, v in scores.items()}
    sum_exp = sum(exp_scores.values())
    probabilities = {k: float(v / sum_exp) for k, v in exp_scores.items()}
    best_category = max(probabilities, key=probabilities.get)
    return best_category, probabilities
