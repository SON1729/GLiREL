"""Server-to-server inference; API secrets never enter the browser."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

def remote_predict(base_url, api_key, example):
    parts = urlparse(base_url)
    if parts.scheme != 'https' and not (parts.scheme == 'http' and parts.hostname in ('127.0.0.1', 'localhost')):
        raise ValueError('GLIREL_API_URL cần HTTPS (hoặc localhost khi chạy trên máy).')
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = Request(base_url.rstrip('/') + '/predict', data=json.dumps({'example': example}).encode(), headers=headers, method='POST')
    try:
        with urlopen(request, timeout=240) as response:
            raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError('Trace trả về vượt 8 MB.')
        return json.loads(raw)
    except HTTPError as exc:
        detail = exc.read(4096).decode(errors='replace')
        try:
            detail = json.loads(detail).get('detail', detail)
        except ValueError:
            pass
        raise ValueError(f'Máy chủ GLiREL trả mã {exc.code}: {str(detail)[:600]}') from None
    except (URLError, TimeoutError) as exc:
        raise ValueError('Máy chủ chưa trả trace; có thể đang khởi động/tải model. Thử lại khi backend sẵn sàng.') from exc
