"""
app/models/__init__.py
"""

from app.models.user import User
from app.models.analysis import Analysis, AnalysisType, SeverityLevel
from app.models.consultation import (
    Dermatologist,
    Appointment,
    Consultation,
    DoctorNote,
    Prescription,
    MeetingRoom,
    AppointmentStatus,
)

__all__ = [
    "User",
    "Analysis",
    "AnalysisType",
    "SeverityLevel",
    "Dermatologist",
    "Appointment",
    "Consultation",
    "DoctorNote",
    "Prescription",
    "MeetingRoom",
    "AppointmentStatus",
]
