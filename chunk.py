import re
from deepdoc.parser.pdf_parser import PlainParser
from nlp import rag_tokenizer, naive_merge, tokenize_table, tokenize_chunks
from deepdoc.parser import PdfParser

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse,Response
from pydantic import BaseModel
from typing import Optional
import uvicorn

from PIL import Image
from io import BytesIO
import base64

class Pdf(PdfParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):
        callback(0.1,msg="OCR is running...")
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(0.2,msg="OCR finished")

        self._layouts_rec(zoomin)
        callback(0.63, "Layout analysis finished.")
        self._table_transformer_job(zoomin)
        callback(0.65, "Table analysis finished.")
        self._text_merge()
        callback(0.67, "Text merging finished")
        tbls = self._extract_table_figure(True, zoomin, True, True)
        #self._naive_vertical_merge()
        self._concat_downward()
        #self._filter_forpages()

        return [(b["text"], self._line_tag(b, zoomin))
                for b in self.boxes], tbls


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
        Supported file formats are docx, pdf, excel, txt.
        This method apply the naive ways to chunk files.
        Successive text will be sliced into pieces using 'delimiter'.
        Next, these successive pieces are merge into chunks whose token number is no more than 'Max token number'.
    """

    eng = lang.lower() == "english"  # is_english(cks)
    parser_config = kwargs.get(
        "parser_config", {
            "chunk_token_num": 128, "delimiter": "\n!?。；！？", "layout_recognize": True})
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])
    res = []
    pdf_parser = None
    sections = []

    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        pdf_parser = Pdf(
        ) if parser_config.get("layout_recognize", True) else PlainParser()
        sections, tbls = pdf_parser(filename if not binary else binary,
                                    from_page=from_page, to_page=to_page, callback=callback)
        res = tokenize_table(tbls, doc, eng)

    else:
        raise NotImplementedError(
            "file type not supported yet(pdf supported)")


    chunks = naive_merge(
        sections, parser_config.get(
            "chunk_token_num", 128), parser_config.get(
            "delimiter", "\n!?。；！？"))

    res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))
    return res

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


app = FastAPI()

class ChunkRequest(BaseModel):
    filename: str
    from_page: Optional[int] = 0
    to_page: Optional[int] = 100000
    lang: Optional[str] = "Chinese"
    parser_config: Optional[dict] = {}

@app.post("/chunk")
async def chunk_endpoint(file: UploadFile = File(...), from_page: int = Form(0), to_page: int = Form(100000), lang: str = Form("Chinese")):
    filename = file.filename
    binary = await file.read()

    def callback(prog=None, msg=None):
        print(f"Progress: {prog}, Message: {msg}")

    result = chunk(filename, binary, from_page, to_page, lang, callback=callback)

    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    for i in result:
        print(i)

    for item in result:
        if 'image' in item:
            item['image'] = encode_image(item['image'])

    return JSONResponse(content={"result":result})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
