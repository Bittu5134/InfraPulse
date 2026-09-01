import markdown
import bleach
from fastapi.templating import Jinja2Templates
from app.config import BASE_DIR

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

ALLOWED_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'code', 'pre', 'blockquote', 'a', 'table', 'thead', 'tbody', 'tr', 'th', 'td'
]
ALLOWED_ATTRS = {'a': ['href', 'title', 'target']}

def render_markdown(text: str) -> str:
    if not text:
        return ""
    html = markdown.markdown(text, extensions=['fenced_code', 'tables', 'nl2br'])
    clean_html = bleach.clean(html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS)
    return clean_html

templates.env.filters["render_md"] = render_markdown
