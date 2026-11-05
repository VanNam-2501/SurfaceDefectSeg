from pathlib import Path
import re

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs" / "BAO_CAO_TONG_QUAN_KHOA_LUAN_ONG_HUT.md"
OUTPUT = ROOT / "submission" / "01_report" / "Bao_cao_tong_quan_khoa_luan_phan_loai_khuyet_tat_ong_hut.docx"
SCREENSHOT = ROOT / "artifacts" / "b1_stream" / "gate_after_roi.jpg"


def set_cell_border(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn("w:" + edge))
        if node is None:
            node = OxmlElement("w:" + edge)
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), "D9E0E8")


def set_cell_margins(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    margins = OxmlElement("w:tcMar")
    for side, value in (("top", 95), ("start", 105), ("bottom", 95), ("end", 105)):
        node = OxmlElement("w:" + side)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    tc_pr.append(margins)


def add_runs(p, text):
    for piece in re.split(r"(\*\*.*?\*\*|`[^`]+`)", text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            p.add_run(piece[2:-2]).bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            r = p.add_run(piece[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(8.6)
        else:
            p.add_run(piece)


def parse_table(lines):
    rows = []
    for line in lines:
        cells = [item.strip() for item in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells):
            continue
        rows.append(cells)
    return rows


def add_table(doc, rows):
    if not rows:
        return
    count = max(map(len, rows))
    table = doc.add_table(rows=1, cols=count)
    table.autofit = True
    for ri, values in enumerate(rows):
        cells = table.rows[0].cells if ri == 0 else table.add_row().cells
        for ci, cell in enumerate(cells):
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.02
            add_runs(p, values[ci] if ci < len(values) else "")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_border(cell)
            set_cell_margins(cell)
            for run in p.runs:
                run.font.size = Pt(8.1)
                if ri == 0:
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            if ri == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), "17365D")
                cell._tc.get_or_add_tcPr().append(shd)
            elif ri % 2 == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), "F3F6FA")
                cell._tc.get_or_add_tcPr().append(shd)
        if ri == 0:
            trpr = table.rows[0]._tr.get_or_add_trPr()
            repeat = OxmlElement("w:tblHeader")
            repeat.set(qn("w:val"), "true")
            trpr.append(repeat)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_page_field(p):
    r = p.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    r._r.extend([begin, instr, end])


def main():
    source_text = SOURCE.read_text(encoding="utf-8")
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin, sec.bottom_margin = Inches(.72), Inches(.72)
    sec.left_margin, sec.right_margin = Inches(.82), Inches(.82)
    sec.header_distance, sec.footer_distance = Inches(.32), Inches(.35)
    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Aptos")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos")
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(35, 43, 52)
    normal.paragraph_format.space_after = Pt(5.5)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in (("Title", 23), ("Heading 1", 16), ("Heading 2", 12.5)):
        style = doc.styles[name]
        style.font.name = "Aptos Display"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Aptos Display")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Aptos Display")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(13 if name == "Heading 1" else 8)
        style.paragraph_format.space_after = Pt(5)
        if name == "Title" and style._element.pPr is not None:
            borders = style._element.pPr.find(qn("w:pBdr"))
            if borders is not None:
                style._element.pPr.remove(borders)
    h = sec.header.paragraphs[0]
    h.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = h.add_run("BÁO CÁO NGHIÊN CỨU  |  PHÂN LOẠI KHUYẾT TẬT ỐNG HÚT")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(85, 97, 110)
    f = sec.footer.paragraphs[0]
    f.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rr = f.add_run("Báo cáo theo hiện trạng dự án  •  ")
    rr.font.size = Pt(8)
    rr.font.color.rgb = RGBColor(95, 105, 116)
    add_page_field(f)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(70)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("BÁO CÁO TỔNG QUAN ĐỒ ÁN")
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(23, 54, 93)
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(16)
    title.paragraph_format.space_after = Pt(19)
    title.add_run("Nghiên cứu kết hợp đặc trưng toàn cục và cục bộ trong mạng CNN cho phân loại khuyết tật ống hút")
    if title._p.pPr is not None:
        borders = title._p.pPr.find(qn("w:pBdr"))
        if borders is not None:
            title._p.pPr.remove(borders)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Bộ dữ liệu ảnh hộp đồ uống và nguyên mẫu kiểm tra trên băng chuyền")
    run.font.size = Pt(12.5)
    run.font.color.rgb = RGBColor(70, 80, 90)
    for text in ("Sinh viên: ........................................................", "Mã số sinh viên: ...............................................", "Giảng viên hướng dẫn: ........................................", "Khoa / Ngành: ..................................................", "Năm học: 2026"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(11)
        p.add_run(text)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(22)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("BẢN BÁO CÁO TỔNG QUAN THEO HIỆN TRẠNG DỰ ÁN")
    r.bold = True
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(100, 105, 110)
    doc.add_page_break()

    lines = source_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
            i += 1
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(doc, parse_table(table_lines))
            continue
        if line.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(.2)
            p.paragraph_format.space_after = Pt(7)
            r = p.add_run("\n".join(code))
            r.font.name = "Consolas"
            r.font.size = Pt(8.2)
            i += 1
            continue
        if line.startswith("\\["):
            formula = []
            i += 1
            while i < len(lines) and not lines[i].startswith("\\]"):
                formula.append(lines[i].strip())
                i += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(" ".join(formula))
            r.font.name = "Cambria Math"
            r.font.size = Pt(10)
            i += 1
            continue
        if line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(2.5)
            add_runs(p, line[2:])
            i += 1
            continue
        if re.match(r"^\d+\. ", line):
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_after = Pt(2.5)
            add_runs(p, re.sub(r"^\d+\. ", "", line))
            i += 1
            continue
        if line.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(.2)
            add_runs(p, line[1:].strip())
            for r in p.runs:
                r.italic = True
            i += 1
            continue
        image_match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            image_path = (SOURCE.parent / image_match.group(2)).resolve()
            if image_path.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(image_path), width=Inches(6.0))
                caption = doc.add_paragraph(image_match.group(1))
                caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in caption.runs:
                    run.italic = True
                    run.font.size = Pt(9)
                i += 1
                continue
        p = doc.add_paragraph()
        add_runs(p, line)
        i += 1

    if SCREENSHOT.exists():
        doc.add_page_break()
        doc.add_heading("Phụ lục D. Ảnh chụp demo tại sự kiện ROI", level=1)
        doc.add_paragraph("Hình minh họa trạng thái mô phỏng sau khi sản phẩm đi qua ROI. Đây là bằng chứng luồng giao diện và sự kiện, không phải đánh giá độ chính xác cuối.")
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(SCREENSHOT), width=Inches(6.35))
        p = doc.add_paragraph("Hình D.1. Demo băng chuyền canvas và sự kiện TensorRT.")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].italic = True
        p.runs[0].font.size = Pt(9)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
