"""
app/schemas/consultation.py

Pydantic v2 validation schemas for Dermatologist Consultation endpoints.
"""

from datetime import date as dt_date, datetime
from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional


# ── Dermatologist Schemas ──────────────────────────────────────────────────────

class DermatologistCreateUpdateRequest(BaseModel):
    specialization: str = Field(..., examples=["General Dermatology & Skin Cancer"])
    biography: Optional[str] = Field(None, examples=["Board-certified skin cancer specialist..."])
    experience_years: int = Field(..., ge=0, examples=[10])
    languages: str = Field(..., description="Comma-separated language list", examples=["English, Spanish"])
    availability: List[str] = Field(..., description="List of time slot strings", examples=[["9:00 AM", "10:30 AM"]])
    price: float = Field(..., ge=0, examples=[150.0])
    profile_picture_url: Optional[str] = None


class DermatologistResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    specialization: str
    biography: Optional[str]
    experience_years: int
    rating: float
    languages: str
    availability: List[str]
    price: float
    is_approved: bool
    profile_picture_url: Optional[str]
    created_at: datetime
    doctor_name: Optional[str] = None
    doctor_email: Optional[str] = None


# ── Appointment Schemas ────────────────────────────────────────────────────────

class AppointmentCreateRequest(BaseModel):
    doctor_id: int = Field(..., examples=[2])
    date: dt_date = Field(..., examples=["2026-06-15"])
    time_slot: str = Field(..., examples=["10:00 AM"])
    reason: str = Field(..., examples=["Mole has grown significantly."])


class AppointmentResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    patient_id: int
    patient_name: Optional[str] = None
    patient_email: Optional[str] = None
    doctor_id: int
    doctor_name: Optional[str] = None
    date: dt_date
    time_slot: str
    reason: str
    status: str
    created_at: datetime


class AppointmentStatusUpdate(BaseModel):
    status: str = Field(..., examples=["accepted", "rejected", "completed"])


# ── Consultation & Room Schemas ────────────────────────────────────────────────

class ConsultationResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    appointment_id: int
    status: str
    created_at: datetime


class MeetingRoomResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    meeting_id: str
    appointment_id: int
    doctor_id: int
    patient_id: int
    created_at: datetime


# ── Doctor Notes & Prescriptions Schemas ────────────────────────────────────────

class DoctorNoteCreateRequest(BaseModel):
    notes: str = Field(..., examples=["Lesion is symmetrical, patient is advised to self-monitor."])


class DoctorNoteResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    appointment_id: int
    doctor_id: int
    patient_id: int
    notes: str
    created_at: datetime


class PrescriptionCreateRequest(BaseModel):
    prescription_text: str = Field(..., examples=["Apply topical cream twice daily."])
    pdf_url: Optional[str] = None


class PrescriptionResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    appointment_id: int
    doctor_id: int
    patient_id: int
    prescription_text: str
    pdf_url: Optional[str]
    created_at: datetime


# ── Admin & Analytics Schemas ──────────────────────────────────────────

class PlatformUserResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    email: EmailStr
    full_name: str
    role: str
    is_active: bool
    created_at: datetime


class PlatformAnalyticsResponse(BaseModel):
    total_users: int
    total_patients: int
    total_dermatologists: int
    total_appointments: int
    appointments_by_status: dict
    completed_consultations: int
