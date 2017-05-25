from __future__ import absolute_import, print_function, unicode_literals

import os
import six
import re

from nbformat.v4.rwbase import NotebookReader
from nbformat.v4.nbbase import (
    new_code_cell, new_raw_cell,
    new_notebook, new_markdown_cell
    )
from nbconvert.preprocessors import ExecutePreprocessor

try:
    # Py3k
    from html.parser import HTMLParser
except ImportError:
    # Py2.7
    from HTMLParser import HTMLParser

from pelican import signals
from pelican.readers import MarkdownReader, HTMLReader, BaseReader

from .ipynb import get_html_from_notebook, fix_css

from markdown import Markdown

def register():
    """
    Register the new "py" reader
    """
    def add_reader(arg):
        arg.settings["READERS"]["py"] = IPythonPy
    signals.initialized.connect(add_reader)

_encoding_declaration_re = re.compile(r"^#.*coding[:=]\s*([-\w.]+)")


class PyReader(NotebookReader):

    def reads(self, s, **kwargs):
        return self.to_notebook(s, **kwargs)

    def to_notebook(self, s, **kwargs):
        lines = s.splitlines()
        cells = []
        cell_lines = []
        kwargs = {}
        state = u'codecell'
        self.metadata = {}
        for line in lines:
            if line.startswith(u'# <nbformat>') or _encoding_declaration_re.match(line):
                pass
            elif line.startswith(u'# <metacell>'):
                cell = self.new_cell(state, cell_lines, **kwargs)
                if cell is not None:
                    cells.append(cell)
                state = u'metacell'
                cell_lines = []
                kwargs = {}
            elif line.startswith(u'# <codecell>'):
                cell = self.new_cell(state, cell_lines, **kwargs)
                if cell is not None:
                    cells.append(cell)
                state = u'codecell'
                cell_lines = []
                kwargs = {}
            elif line.startswith(u'# <markdowncell>'):
                cell = self.new_cell(state, cell_lines, **kwargs)
                if cell is not None:
                    cells.append(cell)
                state = u'markdowncell'
                cell_lines = []
                kwargs = {}
            # VERSIONHACK: plaintext -> raw
            elif line.startswith(u'# <rawcell>'):
                cell = self.new_cell(state, cell_lines, **kwargs)
                if cell is not None:
                    cells.append(cell)
                state = u'rawcell'
                cell_lines = []
                kwargs = {}
            else:
                cell_lines.append(line)
        if cell_lines:
            cell = self.new_cell(state, cell_lines)
            if cell is not None:
                cells.append(cell)

        # Unwrap packed lists
        metadata = {}
        for k, v in self.metadata.items():
            if k == 'authors':
                metadata[k] = v
            else:
                metadata[k] = v[0]
        nb = new_notebook(cells=cells, metadata=metadata)

        return nb

    def new_cell(self, state, lines, **kwargs):
        if state == u'codecell':
            input = u'\n'.join(lines)
            input = input.strip(u'\n')
            if input:
                return new_code_cell(source=input)
        elif state == u'metacell':
            text = self._remove_comments(lines)
            md = Markdown(extensions=['markdown.extensions.meta'])
            md.convert(text)
            if hasattr(md, 'Meta'):
                self.metadata.update(md.Meta)
            return None
        elif state == u'markdowncell':
            text = self._remove_comments(lines)
            if text:
                return new_markdown_cell(source=text)
        elif state == u'rawcell':
            text = self._remove_comments(lines)
            if text:
                return new_raw_cell(source=text)

    def _remove_comments(self, lines):
        new_lines = []
        for line in lines:
            if line.startswith(u'#'):
                new_lines.append(line[2:])
            else:
                new_lines.append(line)
        text = u'\n'.join(new_lines)
        text = text.strip(u'\n')
        return text

    def split_lines_into_blocks(self, lines):
        if len(lines) == 1:
            yield lines[0]
            raise StopIteration()
        import ast
        source = '\n'.join(lines)
        code = ast.parse(source)
        starts = [x.lineno-1 for x in code.body]
        for i in range(len(starts)-1):
            yield '\n'.join(lines[starts[i]:starts[i+1]]).strip('\n')
        yield '\n'.join(lines[starts[-1]:]).strip('\n')


class IPythonPy(BaseReader):
    """
    Extend the Pelican.BaseReader to `.ipynb` files can be recognized
    as a markup language:

    Setup:

    `pelicanconf.py`:
    ```
    MARKUP = ('md', 'ipynb', 'py')
    ```
    """
    enabled = True
    file_extensions = ['py']

    def read(self, filepath):
        metadata = {}
        metadata['ipython'] = True

        # Files
        filedir = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        metadata_filename = os.path.splitext(filename)[0] + '.py-meta'
        metadata_filepath = os.path.join(filedir, metadata_filename)

        reader = PyReader()
        with open(filepath) as f:
            body = f.read()
            nb = reader.reads(body)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python2')
        ep.preprocess(nb, {'metadata': {'path': filedir}})

        if os.path.exists(metadata_filepath):
            # Metadata is on a external file,
            # process using Pelican MD Reader
            md_reader = MarkdownReader(self.settings)
            _content, metadata = md_reader.read(metadata_filepath)
        else:
            notebook_metadata = nb.metadata
            print(notebook_metadata)

            # Change to standard pelican metadata
            for key, value in notebook_metadata.items():
                key = key.lower()
                if key in ("title", "date", "category", "tags", "slug", "author", "summary"):
                    metadata[key] = self.process_metadata(key, value)

        content, info = get_html_from_notebook(nb)

        # Generate Summary: Do it before cleaning CSS
        if 'summary' not in [key.lower() for key in self.settings.keys()]:
            parser = MyHTMLParser(self.settings, filename)
            if isinstance(content, six.binary_type): # PY2 (str) or PY3 (bytes) to PY2 (unicode) or PY3 (str)
                # unicode_literals makes format() try to decode as ASCII. Enforce decoding as UTF-8.
                content = '<body>{0}</body>'.format(content.decode("utf-8"))
            else:
                # Content already decoded
                content = '<body>{0}</body>'.format(content)
            parser.feed(content)
            parser.close()
            content = parser.body
            if ('IPYNB_USE_META_SUMMARY' in self.settings.keys() and \
              self.settings['IPYNB_USE_META_SUMMARY'] == False) or \
              'IPYNB_USE_META_SUMMARY' not in self.settings.keys():
                metadata['summary'] = parser.summary

        ignore_css = True if 'IPYNB_IGNORE_CSS' in self.settings.keys() else False
        content = fix_css(content, info, ignore_css=ignore_css)
        return content, metadata


class MyHTMLParser(HTMLReader._HTMLParser):
    """
    Custom Pelican `HTMLReader._HTMLParser` to create the summary of the content
    based on settings['SUMMARY_MAX_LENGTH'].

    Summary is stoped if founds any div containing ipython notebook code cells.
    This is needed in order to generate valid HTML for the summary,
    a simple string split will break the html generating errors on the theme.
    The downside is that the summary length is not exactly the specified, it stops at
    completed div/p/li/etc tags.
    """
    def __init__(self, settings, filename):
        HTMLReader._HTMLParser.__init__(self, settings, filename)
        self.settings = settings
        self.filename = filename
        self.wordcount = 0
        self.summary = None

        self.stop_tags = [('div', ('class', 'input')), ('div', ('class', 'output')), ('h2', ('id', 'Header-2'))]
        if 'IPYNB_STOP_SUMMARY_TAGS' in self.settings.keys():
            self.stop_tags = self.settings['IPYNB_STOP_SUMMARY_TAGS']
        if 'IPYNB_EXTEND_STOP_SUMMARY_TAGS' in self.settings.keys():
            self.stop_tags.extend(self.settings['IPYNB_EXTEND_STOP_SUMMARY_TAGS'])

    def handle_starttag(self, tag, attrs):
        HTMLReader._HTMLParser.handle_starttag(self, tag, attrs)

        if self.wordcount < self.settings['SUMMARY_MAX_LENGTH']:
            mask = [stoptag[0] == tag and (stoptag[1] is None or stoptag[1] in attrs) for stoptag in self.stop_tags]
            if any(mask):
                self.summary = self._data_buffer
                self.wordcount = self.settings['SUMMARY_MAX_LENGTH']

    def handle_endtag(self, tag):
        HTMLReader._HTMLParser.handle_endtag(self, tag)

        if self.wordcount < self.settings['SUMMARY_MAX_LENGTH']:
            self.wordcount = len(strip_tags(self._data_buffer).split(' '))
            if self.wordcount >= self.settings['SUMMARY_MAX_LENGTH']:
                self.summary = self._data_buffer


def strip_tags(html):
    """
    Strip html tags from html content (str)
    Useful for summary creation
    """
    s = HTMLTagStripper()
    s.feed(html)
    return s.get_data()


class HTMLTagStripper(HTMLParser):
    """
    Custom HTML Parser to strip HTML tags
    Useful for summary creation
    """
    def __init__(self):
        HTMLParser.__init__(self)
        self.reset()
        self.fed = []

    def handle_data(self, html):
        self.fed.append(html)

    def get_data(self):
        return ''.join(self.fed)
