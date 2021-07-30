import os
from datetime import datetime

from jinja2 import Template

from bbavectors import ROOT


def render_html(
    image, results, output_dir=".",
    template_name="report-template.html"
):
    src = os.path.join(ROOT, "app/templates/", template_name)
    with open(src, 'r') as fp:
        template_string = fp.read()

    template = Template(template_string)
    dst = os.path.join(output_dir, f"report.html")

    with open(dst, "w") as hfile:
        hfile.write(template.render(
            image=image, results=results, zip=zip))

    return dst
