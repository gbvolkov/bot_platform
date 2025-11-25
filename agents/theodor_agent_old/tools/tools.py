from langchain_core.tools import tool
from palimpsest import Palimpsest
import logging

from ..augment_query import (
    get_term_meanings
    , get_abbreviation_meaning
    , tnd_docs
)

def get_term_and_defition_tools(anonymizer: Palimpsest = None):
    MAX_RETRIEVALS = 3

    @tool
    def lookup_term(term: str) -> str:
        """
        Look up the definition of a term in the reference dictionary.

        This tool is designed to retrieve the meaning of a full term 
        from a predefined reference source. 
        All terms in the reference are stored in singular nominative case. 

        The input must strictly follow these conventions:
        - Terms: singular nominative case (e.g., "server", "network", "лизинговая заявка").
            Do not translate term to other language. Keep it as provided by user.

        Args:
            name (str): The term to look up.
                Must match the format and casing conventions of the reference.

        Returns:
            str: The definition or description of the provided term.
        """
        try:
            found_docs = get_term_meanings(tnd_docs=tnd_docs, query=term)
            if found_docs:
                result = "\n\n".join([doc["_doc"].page_content for doc in found_docs[:30]])
                if anonymizer:
                    result = anonymizer.anonimize(result)
                return result
            else:
                return "No matching information found."
        except Exception as e:
            logging.error("Error occured during lookup_term tool calling.\nException: {e}")
            raise e
    
    @tool
    def lookup_abbreviation(abbreviation: str) -> str:
        """
        Look up the definition of an abbreviation in the reference dictionary.

        This tool is designed to retrieve the meaning of 
        an abbreviation from a predefined reference source. 
        All abbreviations in the reference are stored in uppercase. 

        The input must strictly follow these conventions:
        - Abbreviations: uppercase only (e.g., "HTTP", "NASA", "АД").
            Do not translate abbreviations to other language. Keep it as provided by user.

        Args:
            name (str): The abbreviation to look up.
                Must match the format and casing conventions of the reference.

        Returns:
            str: The definition or description of the provided abbreviation.
        """
        try:
            found_docs = get_abbreviation_meaning(tnd_docs=tnd_docs, query=abbreviation)
            if found_docs:
                result = "\n\n".join([doc.page_content for doc in found_docs[:30]])
                if anonymizer:
                    result = anonymizer.anonimize(result)
                return result
            else:
                return "No matching information found."
        except Exception as e:
            logging.error("Error occured during lookup_abbreviation tool calling.\nException: {e}")
            raise e

    return (lookup_term, lookup_abbreviation)
