import streamlit as st
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from pydantic import BaseModel, Field, validator
from kor import extract_from_documents, from_pydantic, create_extraction_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
import pandas as pd
import requests
import json
import os
from bs4 import BeautifulSoup
from typing import List, Optional


class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_pdf(self):
        pdfloader = PyPDFLoader(self.file_path)
        pages = pdfloader.load_and_split()
        return pages


class formatOutput:
    def __init__(self, gpt_output):
        self.output = gpt_output

    def print_output(self):
        st.json(output)

    def output_table(self):
        table = pd.DataFrame(output['prenupschema'])
        return table

    def output_csv(self):
        table = pd.DataFrame(output['prenupschema'])
        table.to_csv('parsed_prenup_data.csv', index=False)


class PrenupSchema(BaseModel):
    spouse_1: str = Field(
        description="The name of the first spouse mentioned in the prenuptial agreement",
    )
    spouse_2: str = Field(
        description="The name of the second spouse mentioned in the prenuptial agreement",
    )
    agreement_date: Optional[str] = Field(
        description="Date of when the agreement was signed",
    )
    marriage_date: Optional[str] = Field(
        description="Date of when the couple intends to get legally married",
    )
    state: Optional[str] = Field(
        description="State in the United States where this agreement is formalized or state where the couple lives",
    )
    spouse_1_property: Optional[str] = Field(
        description="All property mentioned in the prenuptial agreement that belongs to the first spouse. Includes money, stocks, cash, real estate, vehicles, art, jewelry, etc.",
    )
    spouse_2_property: Optional[str] = Field(
        description="All property mentioned in the prenuptial agreement that belongs to the second spouse. Includes money, stocks, cash, real estate, vehicles, art, jewelry, etc.",
    )


def main():
    st.title("Prenup Schema Parser")

    # Load PDF document
    loader = DocumentLoader("sample_prenups/Premarital-Agreement_sample.pdf")
    pages = loader.load_pdf()

    schema, extraction_validator = from_pydantic(
        PrenupSchema,
        description="Extract key information from a legal prenuptial agreement between two individuals in a relationship prior to marriage",
        examples=[
            (
                "THIS AGREEMENT made this 15 day of November, 2021, by and between JANE SMITH, residing at Boulder Colorado, hereinafter referred to as “Jane” or “the Wife”, and JOHN DOE, residing at Boulder Colorado, hereinafter referred to as “John” or “theHusband. Property of JANE SMITH includes  Bank of America checking account withan approximate balance of $139,500.00, and a 2020 Mazda 3.Property of JOHN DOE includes The real property known as the Texus Ranch, an authentic Monet painting, and 100% of the funds, stocks, bonds and other assets on deposit in any investment, brokerage, money market, stock and retirement accounts standing in the name of JOHN DOE as of 15/11/2021 ;",
                {"spouse_1": "Jane Smith", "spouse_2": "John Doe", "agreement_date": "15/11/2021", "spouse_1_property": "Bank of America checking account with balance of $139,500.00, 2020 Mazda 3", "spouse_2_property": "Texus Ranch, Monet painting, funds, stocks, bonds"},
            )
        ],
        many=True,
    )

    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0,
        max_tokens=2000,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    chain = create_extraction_chain(llm, schema, input_formatter="triple_quotes")

    # Extract information from relevant pages
    relevant_pages = str(pages[0]) + str(pages[20]) + str(pages[21]) + str(pages[22]) + str(pages[23]) + str(pages[24])
    output = chain.predict_and_parse(text=relevant_pages)["data"]

    return_output = formatOutput(output)
    return_output.output_csv()

    # Display output
    return_output.print_output()


if __name__ == '__main__':
    main()
