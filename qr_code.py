# qr_code.py
import qrcode
import base64
from io import BytesIO

def generate_qr_code_base64(link_url):
    qr = qrcode.QRCode(
        version=1,
        box_size=8,
        border=4
    )
    qr.add_data(link_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    base64_str = base64.b64encode(png_bytes).decode("utf-8")

    return f"data:image/png;base64,{base64_str}"
