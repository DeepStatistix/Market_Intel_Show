import qrcode

# URL for the farmer's app or website
url = "https://your-market-intel-website.com"

# Generate the QR code
img = qrcode.make(url)
img.save('static/qr_code.png')
