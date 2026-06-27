"""
app/models/consultation.py

SQLAlchemy ORM models for the Dermatologist Consultation module:
- Dermatologist
- Appointment
- Consultation
- DoctorNote
- Prescription
- MeetingRoom
"""

from datetime import datetime, date
from enum import Enum as PyEnum
from sqlalchemy import Boolean, DateTime, Date, ForeignKey, Integer, String, Text, Float, Numeric, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class AppointmentStatus(str, PyEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"


class Dermatologist(Base):
    __tablename__ = "dermatologists"

    # 1-to-1 relationship with User
    id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    specialization: Mapped[str] = mapped_column(String(255), nullable=False)
    biography: Mapped[str | None] = mapped_column(Text, nullable=True)
    experience_years: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rating: Mapped[float] = mapped_column(Float, default=5.0, nullable=False)
    languages: Mapped[str] = mapped_column(String(255), default="English", nullable=False)  # Comma-separated
    availability: Mapped[str] = mapped_column(Text, default="[]", nullable=False)  # JSON string representing slot lists
    price: Mapped[float] = mapped_column(Float, default=150.0, nullable=False)
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    profile_picture_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", foreign_keys=[id])  # noqa: F821
    appointments: Mapped[list["Appointment"]] = relationship(
        "Appointment", back_populates="dermatologist", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Dermatologist id={self.id} specialty={self.specialization!r}>"


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    doctor_id: Mapped[int] = mapped_column(Integer, ForeignKey("dermatologists.id", ondelete="CASCADE"), index=True, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    time_slot: Mapped[str] = mapped_column(String(50), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default=AppointmentStatus.PENDING, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    patient: Mapped["User"] = relationship("User", foreign_keys=[patient_id])  # noqa: F821
    dermatologist: Mapped[Dermatologist] = relationship("Dermatologist", back_populates="appointments")
    consultations: Mapped[list["Consultation"]] = relationship(
        "Consultation", back_populates="appointment", cascade="all, delete-orphan"
    )
    meeting_rooms: Mapped[list["MeetingRoom"]] = relationship(
        "MeetingRoom", back_populates="appointment", cascade="all, delete-orphan"
    )
    doctor_notes: Mapped[list["DoctorNote"]] = relationship(
        "DoctorNote", back_populates="appointment", cascade="all, delete-orphan"
    )
    prescriptions: Mapped[list["Prescription"]] = relationship(
        "Prescription", back_populates="appointment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Appointment id={self.id} doctor_id={self.doctor_id} status={self.status}>"


class Consultation(Base):
    __tablename__ = "consultations"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    appointment_id: Mapped[int] = mapped_column(Integer, ForeignKey("appointments.id", ondelete="CASCADE"), index=True, nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="scheduled", nullable=False)  # 'scheduled', 'completed', 'active'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    appointment: Mapped[Appointment] = relationship("Appointment", back_populates="consultations")

    def __repr__(self) -> str:
        return f"<Consultation id={self.id} appointment_id={self.appointment_id}>"


class MeetingRoom(Base):
    __tablename__ = "meeting_rooms"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    meeting_id: Mapped[str] = mapped_column(String(255), nullable=False)
    appointment_id: Mapped[int] = mapped_column(Integer, ForeignKey("appointments.id", ondelete="CASCADE"), index=True, nullable=False)
    doctor_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    appointment: Mapped[Appointment] = relationship("Appointment", back_populates="meeting_rooms")

    def __repr__(self) -> str:
        return f"<MeetingRoom id={self.id} meeting_id={self.meeting_id!r}>"


class DoctorNote(Base):
    __tablename__ = "doctor_notes"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    appointment_id: Mapped[int] = mapped_column(Integer, ForeignKey("appointments.id", ondelete="CASCADE"), index=True, nullable=False)
    doctor_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    notes: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    appointment: Mapped[Appointment] = relationship("Appointment", back_populates="doctor_notes")

    def __repr__(self) -> str:
        return f"<DoctorNote id={self.id} appointment_id={self.appointment_id}>"


class Prescription(Base):
    __tablename__ = "prescriptions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    appointment_id: Mapped[int] = mapped_column(Integer, ForeignKey("appointments.id", ondelete="CASCADE"), index=True, nullable=False)
    doctor_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    prescription_text: Mapped[str] = mapped_column(Text, nullable=False)
    pdf_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    appointment: Mapped[Appointment] = relationship("Appointment", back_populates="prescriptions")

    def __repr__(self) -> str:
        return f"<Prescription id={self.id} appointment_id={self.appointment_id}>"


class ContinuousLearningLog(Base):
    __tablename__ = "continuous_learning_logs"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    analysis_id: Mapped[str] = mapped_column(String(255), nullable=False)
    ai_prediction: Mapped[str] = mapped_column(String(255), nullable=False)
    doctor_correction: Mapped[str | None] = mapped_column(String(255), nullable=True)
    consensus_status: Mapped[str] = mapped_column(String(50), nullable=False)  # 'agree', 'partially_agree', 'disagree'
    doctor_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<ContinuousLearningLog id={self.id} analysis_id={self.analysis_id} status={self.consensus_status}>"
