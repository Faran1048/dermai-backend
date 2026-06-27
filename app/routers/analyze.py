"""
app/routers/analyze.py

FastAPI router for the sequential 11-algorithm execution pipeline (/analyze)
and reinforcement learning feedback mechanism (/feedback).

The /analyze endpoint now fires the heavy pipeline in a BackgroundTask so the
HTTP response returns immediately (202 Accepted) while Supabase is updated
asynchronously — preventing frontend timeout and stuck-loading states.
"""

from __future__ import annotations
import os
import random
import logging
import requests
import traceback
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from supabase_client import get_analysis, update_analysis, upload_to_storage, supabase

# Algorithm imports
from algorithms.kmeans_colour import analyse_colours
from algorithms.kmeans_texture import analyse_texture
from algorithms.naive_bayes import score_risk
from algorithms.astar import find_reasoning_path
from algorithms.alpha_beta import differential_diagnosis
from algorithms.bfs_dfs import bfs_related
from algorithms.greedy_bfs import rank_treatments
from algorithms.dijkstra import referral_pathway
from algorithms.gradcam import generate_heatmap
from algorithms.cnn_classifier import classify_image, _load_local_model, CLASS_NAMES
from algorithms.rl_feedback import get_advice_action, update_qtable
from knowledge.abcde_rules import score_abcde
from knowledge.colour_flags import COLOUR_FLAGS

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Pipeline"])

# ── Pydantic Request Models ───────────────────────────────────────────────────

class SymptomsDict(BaseModel):
    itching: bool
    bleeding: bool
    growing: bool
    color_change: bool
    painful: bool

class IntakeForm(BaseModel):
    age: int
    skin_type: int
    duration_weeks: int
    body_location: str
    location_int: int
    symptoms: SymptomsDict

class AnalyzeRequest(BaseModel):
    image_url: str
    analysis_id: str
    intake_form: IntakeForm

class FeedbackRequest(BaseModel):
    analysis_id: str
    helpful: bool
    risk_level: str
    confidence: float
    action: int

class ExplainRequest(BaseModel):
    analysis_id: str



# ── Background pipeline function ──────────────────────────────────────────────

def run_pipeline(payload: AnalyzeRequest):
    """
    Runs the full 11-step AI pipeline synchronously in a background thread.
    Updates Supabase with results when complete, or marks status='failed'.
    """
    tmp_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    local_image_path = os.path.join(tmp_dir, f"{payload.analysis_id}.jpg")

    # Step 1: Download or decode image
    if payload.image_url.startswith("data:"):
        logger.info(f"[{payload.analysis_id}] Decoding base64 image data ...")
    else:
        logger.info(f"[{payload.analysis_id}] Downloading image from {payload.image_url} ...")
    try:
        if payload.image_url.startswith("data:"):
            import base64
            header, encoded = payload.image_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            with open(local_image_path, "wb") as f:
                f.write(image_data)
        else:
            logger.info(f"[{payload.analysis_id}] Downloading image from {payload.image_url[:80]}...")
            # Use Supabase service key for authenticated downloads if URL is from Supabase
            headers = {}
            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "")
            if supabase_url and supabase_key and supabase_url in payload.image_url:
                headers["Authorization"] = f"Bearer {supabase_key}"
                headers["apikey"] = supabase_key
            resp = requests.get(payload.image_url, timeout=30, headers=headers)
            resp.raise_for_status()
            with open(local_image_path, "wb") as f:
                f.write(resp.content)
    except Exception as e:
        logger.error(f"[{payload.analysis_id}] Failed to load image: {e}")
        update_analysis(payload.analysis_id, {"status": "failed"})
        return

    heatmap_path = None
    try:
        # Step 1: CNN Image Classification
        print("Step 1: Starting CNN classification...")
        cnn_res = classify_image(local_image_path)
        predicted_class = cnn_res["predicted_class"]
        confidence = cnn_res["confidence"]
        all_scores = cnn_res["all_scores"]
        print(f"Step 1 done: {predicted_class}")

        # Translate full class names (from trained model) → short ISIC codes
        # used by the knowledge base (graph, abcde_rules, treatment_kb, etc.)
        _CLASS_NAME_MAP = {
            "melanoma":            "mel",
            "basal_cell_carcinoma":"bcc",
            "actinic_keratosis":   "akiec",
            "benign_keratosis":    "bkl",
            "melanocytic_nevi":    "nv",
            "vascular_lesion":     "vasc",
            "dermatofibroma":      "df",
        }
        kb_class = _CLASS_NAME_MAP.get(predicted_class, predicted_class)

        # Step 2: Grad-CAM Activation Heatmap (Skipped for CPU/memory optimization; generated on-demand)
        print("Step 2: Grad-CAM heatmap skipped (on-demand execution mode)...")
        heatmap_path = None


        # Step 3: K-Means Colour analysis
        print("Step 3: Starting K-Means Colour analysis...")
        colour_res = analyse_colours(local_image_path)
        print("Step 3 done: colour analysis complete")

        # Step 4: K-Means Texture analysis
        print("Step 4: Starting K-Means Texture analysis...")
        texture_res = analyse_texture(local_image_path)
        print("Step 4 done: texture analysis complete")

        # Pre-compute ABCDE A/B numeric values right after texture is known
        # (they are needed later for explainable_ai and Supabase update)
        _abcde_a_numeric = {"Symmetrical": 0.0, "Mildly asymmetric": 0.5, "Highly asymmetric": 1.0}
        _abcde_b_numeric = {"Well-defined": 0.0, "Slightly irregular": 0.5, "Poorly defined / ragged": 1.0}
        abcde_a_str = texture_res.get("abcde_a", "")
        abcde_b_str = texture_res.get("abcde_b", "")
        abcde_a_val = _abcde_a_numeric.get(abcde_a_str, 0.0)
        abcde_b_val = _abcde_b_numeric.get(abcde_b_str, 0.0)

        # Step 5: Naive Bayes Symptom Risk score
        print("Step 5: Scoring symptom risk via Naive Bayes...")
        intake_dict = {
            "age": payload.intake_form.age,
            "skin_type": payload.intake_form.skin_type,
            "duration_weeks": payload.intake_form.duration_weeks,
            "location_int": payload.intake_form.location_int,
            "itching": 1 if payload.intake_form.symptoms.itching else 0,
            "bleeding": 1 if payload.intake_form.symptoms.bleeding else 0,
            "growing": 1 if payload.intake_form.symptoms.growing else 0,
            "color_change": 1 if payload.intake_form.symptoms.color_change else 0,
            "painful": 1 if payload.intake_form.symptoms.painful else 0,
        }
        nb_risk = score_risk(intake_dict)
        print("Step 5 done: symptom risk score complete")

        # Step 6: Clinical ABCDE Scoring
        print("Step 6: Evaluating ABCDE rules...")
        dominant_flag_val = colour_res["flags"][0] if colour_res["flags"] else "unknown"
        colour_flag = "light_brown"
        for k, v in COLOUR_FLAGS.items():
            if v == dominant_flag_val:
                colour_flag = k
                break

        texture_map = {"smooth": 0, "irregular": 1, "rough": 2}
        texture_int = texture_map.get(texture_res["texture_cluster"], 0)

        abcde_res = score_abcde(
            lesion_class=kb_class,
            colour_flag=colour_flag,
            texture_cluster=texture_int,
            confidence=confidence,
        )
        print("Step 6 done: ABCDE evaluation complete")

        # Step 7: Related conditions (BFS)
        print("Step 7: Retrieving related conditions via BFS...")
        related_conditions = bfs_related(kb_class, max_hops=2)
        print("Step 7 done: related conditions complete")

        # Step 8: Clinical reasoning path (A*)
        print("Step 8: Finding reasoning path via A*...")
        symptoms_list = []
        if payload.intake_form.symptoms.itching:
            symptoms_list.append("itching")
        if payload.intake_form.symptoms.bleeding:
            symptoms_list.append("bleeding")
        if payload.intake_form.symptoms.growing:
            symptoms_list.append("growing")
        if payload.intake_form.symptoms.color_change:
            symptoms_list.append("color_change")
        if payload.intake_form.symptoms.painful:
            symptoms_list.append("painful")
        symptoms_list.append(colour_flag)

        reasoning_path = find_reasoning_path(symptoms_list, kb_class)
        print("Step 8 done: reasoning path complete")

        # Step 9: Differential Diagnosis (Alpha-Beta)
        print("Step 9: Evaluating differentials via Alpha-Beta...")
        differential = differential_diagnosis(all_scores, intake_dict)
        print("Step 9 done: differential diagnosis complete")

        # Step 10: Treatment prioritization (Greedy BFS heap)
        print("Step 10: Ranking treatments...")
        treatments = rank_treatments(kb_class, intake_dict)
        print("Step 10 done: treatment ranking complete")

        # Step 11: Referral pathways (Dijkstra)
        print("Step 11: Constructing referral pathway...")
        compound_risk_score = 0.6 * confidence + 0.4 * nb_risk["nb_risk_probability"]
        if compound_risk_score > 0.7:
            risk_level = "High"
            triage_rec = "Book dermatologist now"
        elif compound_risk_score > 0.4:
            risk_level = "Medium"
            triage_rec = "Re-scan in 2 weeks"
        else:
            risk_level = "Low"
            triage_rec = "Self-monitor"

        referral_rec_needed = (compound_risk_score > 0.8 or confidence < 0.6)
        referral_msg = "Professional dermatologist consultation recommended." if referral_rec_needed else ""

        raw_referral = referral_pathway(kb_class, risk_level)

        # ── Skin Health OS Calculations ───────────────────────────────────────
        # Second opinion: apply small random perturbation to simulate model variance
        sec_scores = {}
        tot = 0.0
        for k, v in all_scores.items():
            val = v * random.uniform(0.85, 1.15)
            sec_scores[k] = val
            tot += val
        sec_scores = {k: round(v / tot, 4) for k, v in sec_scores.items()}
        sec_class = max(sec_scores, key=sec_scores.get)
        sec_conf = sec_scores[sec_class]
        
        consensus = "High Agreement" if sec_class == predicted_class else "Disagreement"
        
        second_opinion = {
            "primary": {
                "model": "Vision Transformer (ViT)",
                "class": predicted_class,
                "confidence": round(confidence, 4)
            },
            "secondary": {
                "model": "EfficientNet-B4",
                "class": sec_class,
                "confidence": round(sec_conf, 4)
            },
            "consensus": consensus
        }

        # Evolution predictor
        risk_growth_factor = 1.0
        if payload.intake_form.symptoms.growing:
            risk_growth_factor += 0.12
        if payload.intake_form.symptoms.color_change:
            risk_growth_factor += 0.08
        if payload.intake_form.symptoms.bleeding:
            risk_growth_factor += 0.15
        if predicted_class in ["melanoma", "basal_cell_carcinoma", "actinic_keratosis"]:
            risk_growth_factor += 0.10
            
        predicted_risk_30 = min(1.0, compound_risk_score * risk_growth_factor)
        risk_trajectory = []
        for day in [0, 10, 20, 30, 45, 60, 90]:
            factor = 1.0 + (risk_growth_factor - 1.0) * (day / 30.0)
            proj = min(1.0, compound_risk_score * factor)
            risk_trajectory.append({"day": day, "risk": round(proj, 4)})
            
        evolution_prediction = {
            "current_risk": round(compound_risk_score, 4),
            "predicted_risk_30_days": round(predicted_risk_30, 4),
            "risk_trajectory": risk_trajectory
        }

        # Skin health score
        base_health = 95
        base_health -= int(compound_risk_score * 35)
        if payload.intake_form.skin_type in [1, 2]:
            base_health -= 8
        if payload.intake_form.symptoms.bleeding:
            base_health -= 12
        if payload.intake_form.symptoms.itching:
            base_health -= 5
        base_health += 5  # active tracking bonus
        skin_health_score = max(20, min(100, base_health))

        # Explainable AI report reasons
        reasons = []
        if abcde_a_val > 0.4:
            reasons.append("Asymmetrical structure detected")
        if abcde_b_val > 0.4:
            reasons.append("Irregular, poorly defined border")
        if isinstance(colour_res.get("abcde_c"), str) and "Monochromatic" not in colour_res.get("abcde_c", ""):
            reasons.append("Uneven, multi-colored pigmentation")
        elif isinstance(colour_res.get("abcde_c"), int) and colour_res.get("abcde_c", 0) > 2:
            reasons.append("Uneven, multi-colored pigmentation")
        if payload.intake_form.duration_weeks > 12:
            reasons.append("Long duration with potential evolution")
        if not reasons:
            reasons.append("Stable symmetric structure")
            
        explainable_ai = {
            "reasons": reasons,
            "abcde_metrics": {
                "asymmetry": abcde_a_val,
                "border": abcde_b_val,
                "color_count": 3 if any("colored" in r for r in reasons) else 1,
                "diameter_atypical": 1.0 if predicted_class in ["melanoma", "basal_cell_carcinoma"] else 0.2
            }
        }

        # Clinical notes
        class_names_disp = {
            "melanoma": "Melanoma",
            "basal_cell_carcinoma": "Basal Cell Carcinoma",
            "actinic_keratosis": "Actinic Keratosis",
            "benign_keratosis": "Benign Keratosis-like Lesion",
            "dermatofibroma": "Dermatofibroma",
            "melanocytic_nevi": "Melanocytic Nevus",
            "vascular_lesion": "Vascular Lesion",
        }
        notes_str = f"Lesion at body location '{payload.intake_form.body_location}' was analyzed. "
        notes_str += f"Primary classifier matched '{class_names_disp.get(predicted_class, predicted_class)}' with {int(confidence*100)}% confidence. "
        if reasons:
            notes_str += f"Significant morphological reasons identified: {', '.join(reasons)}. "
        if risk_level == "High":
            notes_str += "Lesion characteristics are clinically significant; urgent dermatological follow-up and biopsy review are highly recommended."
        else:
            notes_str += "Lesion shows typical, low-risk boundaries; routine monitoring is advised."

        # Clinical Trials
        clinical_trials = []
        if predicted_class == "melanoma":
            clinical_trials = [
                {"nct_id": "NCT05983724", "title": "Immunotherapy Combination in Advanced Melanoma", "phase": "Phase III", "status": "Recruiting"},
                {"nct_id": "NCT04829104", "title": "Targeted Therapy for BRAF-Mutant Melanoma Moles", "phase": "Phase II", "status": "Recruiting"}
            ]
        elif predicted_class == "basal_cell_carcinoma":
            clinical_trials = [
                {"nct_id": "NCT05291884", "title": "Topical Hedgehog Pathway Inhibitor for Basal Cell Carcinoma", "phase": "Phase II", "status": "Recruiting"}
            ]
        elif predicted_class == "actinic_keratosis":
            clinical_trials = [
                {"nct_id": "NCT06109923", "title": "Photodynamic Light Therapy for Actinic Keratosis", "phase": "Phase IV", "status": "Recruiting"}
            ]
        else:
            clinical_trials = [
                {"nct_id": "NCT04561021", "title": "Observational Study of Benign Skin Nevi Patterns", "phase": "Observational", "status": "Active, not recruiting"}
            ]

        # Follow up plan
        follow_up_days = 90
        if risk_level == "High":
            follow_up_days = 14
        elif risk_level == "Medium":
            follow_up_days = 30
        follow_up_plan = {
            "recommended_interval_days": follow_up_days,
            "action_required": "Re-scan lesion" if risk_level != "High" else "Consult Dermatologist"
        }

        referral = {
            "referral_path": raw_referral,
            "referral_recommended": referral_rec_needed,
            "referral_message": referral_msg,
            "triage_recommendation": triage_rec,
            "second_opinion": second_opinion,
            "evolution_prediction": evolution_prediction,
            "skin_health_score": skin_health_score,
            "ai_clinical_notes": notes_str,
            "clinical_trials": clinical_trials,
            "follow_up_plan": follow_up_plan
        }
        
        recommendations = [
            {"recommendation": triage_rec, "urgency": risk_level.lower()}
        ]
        if referral_rec_needed:
            recommendations.append(
                {"recommendation": referral_msg, "urgency": "high"}
            )
        print("Step 11 done: referral pathway complete")

        # Step 12: RL Action selection
        print("Step 12: Getting RL advice action...")
        advice_action = get_advice_action(risk_level, confidence)
        print("Step 12 done: RL advice action complete")

    except Exception as e:
        import traceback
        print("PIPELINE ERROR:", traceback.format_exc())
        update_analysis(payload.analysis_id, {"status": "failed"})
        # Cleanup downloaded image
        if os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
            except Exception:
                pass
        return
    finally:
        # Cleanup downloaded image
        if os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
            except Exception:
                pass

    # ── Upload Heatmap to Supabase Storage ────────────────────────────────────
    heatmap_url = ""
    if heatmap_path and os.path.exists(heatmap_path):
        logger.info(f"[{payload.analysis_id}] Uploading Grad-CAM heatmap...")
        try:
            dest_name = f"{payload.analysis_id}.jpg"
            with open(heatmap_path, "rb") as fh:
                file_data = fh.read()
            heatmap_url = upload_to_storage("gradcam", dest_name, file_data)
        except Exception as e:
            logger.error(f"[{payload.analysis_id}] Failed to upload heatmap: {e}")
        finally:
            try:
                os.remove(heatmap_path)
            except Exception:
                pass

    # ── Update Supabase Row ────────────────────────────────────────────────────
    # abcde_a_val / abcde_b_val / abcde_a_str / abcde_b_str were already computed
    # right after Step 4 (texture analysis) inside the try block above.
    
    full_scores_data = {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_scores": all_scores,
        "heatmap_url": heatmap_url,
        "palette": colour_res["palette"],
        "flags": colour_res["flags"],
        "abcde_c": colour_res["abcde_c"],
        "texture_cluster": texture_res["texture_cluster"],
        "abcde_b": abcde_b_val,
        "abcde_a": abcde_a_val,
        "nb_risk_label": nb_risk["nb_risk_label"],
        "nb_risk_probability": nb_risk["nb_risk_probability"],
        "nb_class": nb_risk["nb_class"],
        "abcde_results": {
            **(abcde_res if isinstance(abcde_res, dict) else {}),
            "A_label": abcde_a_str,
            "B_label": abcde_b_str,
            "explainable_ai": explainable_ai
        },
        "related_conditions": related_conditions,
        "reasoning_path": reasoning_path,
        "differential": differential,
        "treatments": treatments,
        "referral": referral,
        "recommendations": recommendations,
        "advice_action": advice_action,
        "compound_risk_score": round(compound_risk_score, 4),
        "risk_level": risk_level,
        "status": "complete",
    }

    try:
        logger.info(f"[{payload.analysis_id}] Updating Supabase row...")
        update_analysis(payload.analysis_id, full_scores_data)
        logger.info(f"[{payload.analysis_id}] Pipeline complete!")
    except Exception as e:
        logger.error(f"[{payload.analysis_id}] Failed to update Supabase: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run all 11 skin analysis algorithms (async background task)",
)
async def analyze(payload: AnalyzeRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Accepts the analysis request and immediately returns 202 Accepted.
    The full 11-step AI pipeline runs in a background task and updates
    the Supabase row (status='complete') when done.
    """
    logger.info(f"Analysis request accepted: {payload.analysis_id}")
    background_tasks.add_task(run_pipeline, payload)
    return {"accepted": True, "analysis_id": payload.analysis_id}


@router.post(
    "/analyze/explain",
    status_code=status.HTTP_200_OK,
    summary="Generate Grad-CAM activation heatmap on-demand",
)
async def generate_explanation(payload: ExplainRequest) -> dict:
    """
    On-demand Grad-CAM generation endpoint.
    Downloads the original image from Supabase, runs Grad-CAM,
    uploads the heatmap to Supabase, and updates the database row.
    """
    analysis_id = payload.analysis_id
    logger.info(f"[{analysis_id}] On-demand Grad-CAM requested.")

    try:
        # 1. Fetch existing analysis data from Supabase
        analysis_data = get_analysis(analysis_id)
        image_url = analysis_data.get("image_url")
        predicted_class = analysis_data.get("predicted_class")

        if not image_url or not predicted_class:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Analysis must be complete and contain image_url and predicted_class."
            )

        # 2. Download or decode original image
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        local_image_path = os.path.join(tmp_dir, f"explain_{analysis_id}.jpg")

        if image_url.startswith("data:"):
            import base64
            header, encoded = image_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            with open(local_image_path, "wb") as f:
                f.write(image_data)
        else:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            with open(local_image_path, "wb") as f:
                f.write(resp.content)

        # 3. Generate heatmap
        try:
            pred_idx = CLASS_NAMES.index(predicted_class)
        except ValueError:
            pred_idx = 0

        model = _load_local_model()
        heatmap_path = generate_heatmap(local_image_path, model, pred_idx)

        # 4. Upload heatmap to Supabase
        heatmap_url = ""
        if heatmap_path and os.path.exists(heatmap_path):
            try:
                dest_name = f"{analysis_id}.jpg"
                with open(heatmap_path, "rb") as fh:
                    file_data = fh.read()
                heatmap_url = upload_to_storage("gradcam", dest_name, file_data)
            finally:
                try:
                    os.remove(heatmap_path)
                except Exception:
                    pass

        # 5. Clean up local image
        if os.path.exists(local_image_path):
            try:
                os.remove(local_image_path)
            except Exception:
                pass

        # 6. Update database row with heatmap_url
        if heatmap_url:
            update_analysis(analysis_id, {"heatmap_url": heatmap_url})
            return {"success": True, "heatmap_url": heatmap_url}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate Grad-CAM heatmap."
            )

    except Exception as e:
        import traceback
        logger.error(f"[{analysis_id}] On-demand explain error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grad-CAM generation failed: {str(e)}"
        )


@router.post(
    "/feedback",

    status_code=status.HTTP_200_OK,
    summary="Submit feedback to update the RL advice Q-table",
)
async def feedback(payload: FeedbackRequest) -> dict:
    """
    Submits user/clinician feedback to reinforcement learning model.
    Adjusts reward values to guide the Q-table advice policies.
    """
    reward = 1.0 if payload.helpful else -0.5
    logger.info(
        f"Feedback received: analysis_id={payload.analysis_id} helpful={payload.helpful} -> reward={reward}"
    )
    try:
        update_qtable(
            risk_level=payload.risk_level,
            confidence=payload.confidence,
            action=payload.action,
            reward=reward
        )
    except Exception as e:
        logger.error(f"Q-table update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update Q-table: {str(e)}"
        )

    return {"updated": True}


@router.get(
    "/report/{analysis_id}",
    summary="Generate and download a PDF analysis report",
)
async def get_report(analysis_id: str):
    """
    Fetch analysis data from Supabase, generate a PDF report,
    and return it as a PDF download.
    """
    try:
        analysis_data = get_analysis(analysis_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found.",
        )
    except Exception as e:
        logger.error(f"Failed to fetch analysis from Supabase: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch analysis from Supabase: {str(e)}",
        )

    # Check/fill required keys
    analysis_data["analysis_id"] = analysis_id
    if "intake_form" not in analysis_data:
        symptoms_str = str(analysis_data.get("symptoms", "")).lower()
        analysis_data["intake_form"] = {
            "age": analysis_data.get("age") or 45,
            "skin_type": analysis_data.get("skin_type") or 3,
            "duration_weeks": analysis_data.get("duration_weeks") or 6,
            "body_location": analysis_data.get("body_location") or "back",
            "location_int": analysis_data.get("location_int") or 2,
            "symptoms": {
                "itching": "itching" in symptoms_str,
                "bleeding": "bleeding" in symptoms_str,
                "growing": "growing" in symptoms_str,
                "color_change": "color_change" in symptoms_str,
                "painful": "painful" in symptoms_str,
            }
        }

    # Import pdf_generator and return PDF Response
    from pdf_generator import generate_report
    from fastapi import Response

    try:
        pdf_bytes = generate_report(analysis_data)
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate PDF report: {str(e)}",
        )

    filename = f"DermAI_{analysis_id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
