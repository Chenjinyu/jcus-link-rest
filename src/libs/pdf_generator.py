"""
Generate PDF resumes from JSON data using ReportLab.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    ListFlowable,
    ListItem,
    KeepTogether,
    HRFlowable,
    Table,
    TableStyle,
    Flowable
)


def _load_resume_data(json_path: str | Path) -> dict[str, Any]:
    """
    Load resume data from a JSON file.

    Args:
        json_path: Path to the JSON resume data file.

    Returns:
        Parsed resume data dictionary.
    """
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _format_location(entry: dict[str, Any]) -> str:
    """
    Format location from city/state fields.

    Args:
        entry: Resume entry containing city/state fields.

    Returns:
        Formatted location string.
    """
    city = entry.get("city")
    state = entry.get("state_or_country")
    if city and state:
        return f"{city}, {state}"
    return city or state or ""


def _format_date_range(entry: dict[str, Any]) -> str:
    """
    Format a date range from start/end fields.

    Args:
        entry: Resume entry containing start/end date fields.

    Returns:
        Formatted date range string.
    """
    start = entry.get("start_date")
    end = entry.get("end_date")
    if start and end:
        return f"{start} - {end}"
    return start or end or ""


def _build_skill_keywords(text: str | None) -> str | None:
    """
    Normalize skill keywords text if present.

    Args:
        text: Raw skill keyword text.

    Returns:
        Normalized skill keyword text or None.
    """
    if not text:
        return None
    return text.strip()


def _register_font(font_name: str, font_path: Path) -> bool:
    """
    Register a TrueType font if the file exists.

    Args:
        font_name: ReportLab font name to register.
        font_path: Path to the TrueType font file.

    Returns:
        True if the font was registered, otherwise False.
    """
    if not font_path.exists():
        return False
    pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
    return True


def _resolve_output_path(data: dict[str, Any], output_path: str | Path) -> Path:
    """
    Resolve the output PDF path using the resume name.

    Args:
        data: Resume data dictionary.
        output_path: Output file path or directory.

    Returns:
        Resolved PDF output path.
    """
    name = str(data.get("basic_info", {}).get("name") or "resume").strip() or "resume"
    if not str(output_path).strip():
        return Path(f"{name}'s resume.pdf")

    path = Path(output_path)
    filename = f"{name}_resume.pdf"
    if path.suffix.lower() == ".pdf":
        return path
    if path.exists() and path.is_dir():
        return path / filename
    return path / filename


def _resolve_font_family() -> tuple[str, str, str]:
    """
    Resolve base, bold, and italic font names, preferring Arial.

    Args:
        None.

    Returns:
        Tuple of (base_font, bold_font, italic_font) names.
    """
    arial_paths = [
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path.home() / "Library/Fonts/Arial.ttf",
    ]
    arial_bold_paths = [
        Path("/Library/Fonts/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path.home() / "Library/Fonts/Arial Bold.ttf",
    ]
    arial_italic_paths = [
        Path("/Library/Fonts/Arial Italic.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        Path.home() / "Library/Fonts/Arial Italic.ttf",
    ]
    base_font = "Helvetica"
    bold_font = "Helvetica-Bold"
    italic_font = "Helvetica-Oblique"

    for path in arial_paths:
        if _register_font("Arial", path):
            base_font = "Arial"
            break
    for path in arial_bold_paths:
        if _register_font("Arial-Bold", path):
            bold_font = "Arial-Bold"
            break
    for path in arial_italic_paths:
        if _register_font("Arial-Italic", path):
            italic_font = "Arial-Italic"
            break

    return base_font, bold_font, italic_font


def _format_contact_line(basic_info: dict[str, Any], link_color: str) -> str:
    """
    Build a contact line with colorized email and links.

    Args:
        basic_info: Basic info data from resume JSON.
        link_color: Hex color code for link text.

    Returns:
        ReportLab paragraph markup for the contact line.
    """
    address = basic_info.get("address")
    mobile = basic_info.get("mobile")
    email = basic_info.get("email")
    github = basic_info.get("github")
    linkedin = basic_info.get("linkedin")

    parts: list[str] = []
    for value in [address, mobile]:
        if value:
            parts.append(escape(str(value)))
    for value in [email, github, linkedin]:
        if value:
            safe_value = escape(str(value))
            parts.append(f'<font color="{link_color}">{safe_value}</font>')
    return " | ".join(parts)


def _build_contact_table(
    basic_info: dict[str, Any],
    meta: ParagraphStyle,
    link_color: str,
    table_width: float,
) -> Table:
    """
    Build a two-row contact table with aligned columns and clickable links.

    Args:
        basic_info: Basic info data from resume JSON.
        meta: Paragraph style for contact text.
        link_color: Hex color code for link text.
        table_width: Width available for the table.

    Returns:
        ReportLab Table for contact information.
    """
    address = escape(str(basic_info.get("address") or ""))
    mobile = escape(str(basic_info.get("mobile") or ""))
    email = basic_info.get("email")
    github = basic_info.get("github")
    linkedin = basic_info.get("linkedin")

    email_cell = ""
    if email:
        email_safe = escape(str(email))
        email_cell = (
            f'<link href="mailto:{email_safe}">'
            f'<font color="{link_color}">{email_safe}</font>'
            "</link>"
        )
    github_cell = ""
    if github:
        github_safe = escape(str(github))
        github_cell = (
            f'<link href="{github_safe}">'
            f'<font color="{link_color}">{github_safe}</font>'
            "</link>"
        )
    linkedin_cell = ""
    if linkedin:
        linkedin_safe = escape(str(linkedin))
        linkedin_cell = (
            f'<link href="{linkedin_safe}">'
            f'<font color="{link_color}">{linkedin_safe}</font>'
            "</link>"
        )

    row_one = [
        Paragraph(address, meta),
        Paragraph(mobile, meta),
        Paragraph(email_cell, meta),
    ]
    row_two = [
        Paragraph(github_cell, meta),
        Paragraph("", meta),
        Paragraph(linkedin_cell, meta),
    ]
    table = Table(
        [row_one, row_two],
        colWidths=[table_width * 0.34, table_width * 0.2, table_width * 0.46],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (0, 0), "LEFT"),
                ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                ("ALIGN", (0, 1), (0, 1), "LEFT"),
                ("ALIGN", (2, 1), (2, 1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _build_professional_experience(
    story: list[Any],
    experience: list[dict[str, Any]],
    section: ParagraphStyle,
    subheader: ParagraphStyle,
    meta: ParagraphStyle,
    normal: ParagraphStyle,
    bold_font: str,
) -> None:
    """
    Append professional experience entries to the story.

    Args:
        story: Flowable list to append into.
        experience: Professional experience entries.
        section: Section title style.
        subheader: Subheader style.
        meta: Metadata text style.
        normal: Normal body text style.
        bold_font: Font name for bold text.

    Returns:
        None.
    """
    if not experience:
        return

    story.append(Paragraph("PROFESSIONAL EXPERIENCE", section))
    story.append(HRFlowable(width="100%", thickness=cast(int, 0.8), color=colors.HexColor("#d0d0d0")))
    for index, entry in enumerate(experience):
        company = entry.get("company")
        job_title = entry.get("job_title")
        location = _format_location(entry)
        dates = _format_date_range(entry)
        header_parts = [part for part in [company, job_title] if part]
        if header_parts:
            story.append(Paragraph(" | ".join(header_parts), subheader))
        meta_parts = [part for part in [location, dates] if part]
        if meta_parts:
            story.append(Paragraph(" | ".join(meta_parts), meta))

        skills = _build_skill_keywords(entry.get("skill_keywords"))
        if skills:
            skill_style = ParagraphStyle(
                "SkillKeywords",
                parent=normal,
                fontName=bold_font,
                fontSize=9,
            )
            skill_table = Table(
                [[Paragraph(skills, skill_style)]],
                colWidths=["100%"],
            )
            skill_table.setStyle(
                TableStyle(
                    [
                        ("BOX", (0, 0), (-1, -1), 1, colors.black),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            story.append(skill_table)

        positions = entry.get("positions", []) or []
        for position in positions:
            role = position.get("role")
            if role:
                story.append(Paragraph(role, subheader))
            projects = position.get("projects", []) or []
            for project in projects:
                summary = project.get("project_summary")
                details = project.get("project_details", []) or []
                if summary:
                    sub_bullets: list[Any] = []
                    if details:
                        sub_bullets = [
                            ListItem(Paragraph(detail, normal), leftIndent=12)
                            for detail in details
                            if detail
                        ]
                    # where bullet_item_flowables is created
                    bullet_item_flowables: list[Flowable] = [Paragraph(f"<b>{summary}</b>", normal)]
                    if sub_bullets:
                        bullet_item_flowables.append(
                            ListFlowable(
                                sub_bullets,
                                bulletType="bullet",
                                bulletChar="o",
                                leftIndent=16,
                            )
                        )
                    bullets = ListFlowable(
                        [ListItem(bullet_item_flowables, leftIndent=12)],
                        bulletType="bullet",
                        leftIndent=12,
                    )
                    story.append(KeepTogether(bullets))
                elif details:
                    bullets = ListFlowable(
                        [
                            ListItem(Paragraph(detail, normal), leftIndent=12)
                            for detail in details
                            if detail
                        ],
                        bulletType="bullet",
                        leftIndent=12,
                    )
                    story.append(KeepTogether(bullets))

        story.append(Spacer(1, 4))
        if index < len(experience) - 1:
            story.append(
                HRFlowable(width="100%", thickness=cast(int, 0.6), color=colors.HexColor("#d0d0d0"))
            )
            story.append(Spacer(1, 4))


def _build_education(
    story: list[Any],
    education: list[dict[str, Any]],
    section: ParagraphStyle,
    subheader: ParagraphStyle,
    meta: ParagraphStyle,
) -> None:
    """
    Append education entries to the story.

    Args:
        story: Flowable list to append into.
        education: Education entries.
        section: Section title style.
        subheader: Subheader style.
        meta: Metadata text style.

    Returns:
        None.
    """
    if not education:
        return

    story.append(Paragraph("EDUCATION", section))
    story.append(HRFlowable(width="100%", thickness=cast(int, 0.8), color=colors.HexColor("#d0d0d0")))
    for entry in education:
        degree = entry.get("degree")
        institution = entry.get("institution")
        dates = _format_date_range(entry)
        line_parts = [part for part in [degree, institution] if part]
        if line_parts:
            story.append(Paragraph(" | ".join(line_parts), subheader))
        if dates:
            story.append(Paragraph(dates, meta))
    story.append(Spacer(1, 4))


def _build_certifications(
    story: list[Any],
    certifications: list[dict[str, Any]],
    section: ParagraphStyle,
    subheader: ParagraphStyle,
    meta: ParagraphStyle,
) -> None:
    """
    Append certification entries to the story.

    Args:
        story: Flowable list to append into.
        certifications: Certification entries.
        section: Section title style.
        subheader: Subheader style.
        meta: Metadata text style.

    Returns:
        None.
    """
    if not certifications:
        return

    story.append(Paragraph("LICENSES & CERTIFICATIONS", section))
    story.append(HRFlowable(width="100%", thickness=cast(int, 0.8), color=colors.HexColor("#d0d0d0")))
    for entry in certifications:
        name = entry.get("name")
        issuer = entry.get("issuer")
        entry_type = entry.get("type")
        validation = entry.get("validation_number")
        parts = [part for part in [name, issuer] if part]
        if parts:
            story.append(Paragraph(" | ".join(parts), subheader))
        meta_parts = [part for part in [entry_type, validation] if part]
        if meta_parts:
            story.append(Paragraph(" | ".join(meta_parts), meta))
    story.append(Spacer(1, 4))


def generate_resume_pdf(data: dict[str, Any], output_path: str | Path) -> None:
    """
    Generate a resume PDF from structured data.

    Args:
        data: Resume data dictionary.
        output_path: Output PDF file path.

    Returns:
        None.
    """
    styles = getSampleStyleSheet()
    base_font, bold_font, italic_font = _resolve_font_family()
    normal = styles["BodyText"]
    normal.spaceAfter = 4
    normal.leading = 12
    normal.fontName = base_font
    normal.fontSize = 9.5
    normal.textColor = colors.HexColor("#1f2933")

    header = ParagraphStyle(
        "Header",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=11,
        spaceAfter=4,
        textColor=colors.HexColor("#1a1a1a"),
    )
    section = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName=italic_font,
        fontSize=10,
        spaceBefore=4,
        spaceAfter=4,
    textColor=colors.HexColor("#369bad"),
        underline=True,
    )
    subheader = ParagraphStyle(
        "Subheader",
        parent=styles["Heading3"],
        fontName=bold_font,
        fontSize=10,
        spaceAfter=4,
        textColor=colors.HexColor("#333333"),
    )
    meta = ParagraphStyle(
        "Meta",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4,
    )

    resolved_output = _resolve_output_path(data, output_path)
    doc = SimpleDocTemplate(
        str(resolved_output),
        pagesize=LETTER,
        leftMargin=0.375 * inch,
        rightMargin=0.375 * inch,
        topMargin=0.25 * inch,
        bottomMargin=0.25 * inch,
    )

    story: list[Any] = []
    basic_info = data.get("basic_info", {})

    name = basic_info.get("name", "Resume")
    story.append(Paragraph(name, header))
    story.append(HRFlowable(width="100%", thickness=cast(int, 1.0), color=colors.HexColor("#b0b0b0")))
    story.append(_build_contact_table(basic_info, meta, "#3ac9c7", doc.width))

    _build_professional_experience(
        story,
        data.get("work_experience", []) or [],
        section,
        subheader,
        meta,
        normal,
        bold_font,
    )
    _build_education(
        story,
        data.get("education", []) or [],
        section,
        subheader,
        meta,
    )
    _build_certifications(
        story,
        data.get("licenses_and_certifications", []) or [],
        section,
        subheader,
        meta,
    )

    doc.build(story)


def generate_resume_pdf_from_json(json_path: str | Path, output_path: str | Path | None = None) -> None:
    """
    Load resume data from JSON and generate a PDF file.

    Args:
        json_path: Path to the JSON resume data file.
        output_path: Output PDF file path.

    Returns:
        None.
    """
    data = _load_resume_data(json_path)
    if output_path is None:
        output_path = ""
    generate_resume_pdf(data, output_path)
