import os
#from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.oauth2 import service_account
#from google_auth_httplib2 import AuthorizedHttp

#import httplib2

class GoogleSheetsManager:
    def __init__(self, credentials_file, spreadsheet_id):
        self.credentials_file = credentials_file
        self.spreadsheet_id = spreadsheet_id
        self.service = self._create_service()

    def _create_service(self):
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_file, 
            scopes=['https://www.googleapis.com/auth/spreadsheets']
        )
        # Create an HTTP client with SSL verification disabled
        #http_client = httplib2.Http(disable_ssl_certificate_validation=True)
        # Wrap the Http object with AuthorizedHttp using your credentials.
        #authed_http = AuthorizedHttp(creds, http=http_client)

        return build('sheets', 'v4', credentials=creds) #build('sheets', 'v4', http=authed_http)

    def append_row(self, values, sheet_range='feedback'):
        sheet = self.service.spreadsheets().values()
        return sheet.append(
            spreadsheetId=self.spreadsheet_id,
            range=sheet_range,
            valueInputOption='USER_ENTERED',
            insertDataOption='INSERT_ROWS',
            body={'values': [values]},
        ).execute()
    
    def process_answers(self, processor, answers_range='answers', status='1', processed_range='moderation'):
        # Get the values from the 'answers' sheet.

        cprocessed = 0
        sheet_values = self.service.spreadsheets().values()
        result = sheet_values.get(spreadsheetId=self.spreadsheet_id, range=answers_range).execute()
        values = result.get('values', [])
        if not values:
            print("No data found.")
            return cprocessed

        headers = values[0]
        try:
            status_index = headers.index('status')
        except ValueError as e:
            raise Exception("Status column not found in headers") from e

        # Get the sheetId for the 'answers' sheet (needed for deletion).
        answers_sheet_id = self.get_sheet_id(answers_range)

        # deleted_count keeps track of how many rows have been removed already,
        # which is needed to compute the effective row index for deletion.
        deleted_count = 0

        # Iterate over data rows (starting at row 2 because row 1 holds headers).
        for row_no, row in enumerate(values[1:], start=2):
            # Ensure that the row has enough columns and check its status.
            if len(row) > status_index and row[status_index] == status:
                record = dict(zip(headers, row))
                question = record.get("user_question", "NA")
                answer = record.get("user_answer", "NA")

                if processor(question, answer):
                    cprocessed = cprocessed + 1
                    # Update the row's status to '2' (as a string to be consistent)
                    row[status_index] = '2'

                    # Append the updated row to the 'processed' sheet.
                    # (This uses the append method so the row is added to the bottom.)
                    self.append_row(row, sheet_range=processed_range)
                    #append_body = {"values": [row]}
                    #self.service.spreadsheets().values().append(
                    #    spreadsheetId=self.spreadsheet_id,
                    #    range=f"{processed_range}!A1",
                    #    valueInputOption="RAW",
                    #    body=append_body
                    #).execute()

                    # Compute the effective row index in the sheet.
                    # The API’s deleteDimension expects 0-indexed row numbers.
                    # (row_no is 1-indexed; subtract one plus the number of previously deleted rows)
                    effective_row_index = row_no - 1 - deleted_count

                    # Delete the processed row from the 'answers' sheet.
                    delete_request = {
                        "requests": [
                            {
                                "deleteDimension": {
                                    "range": {
                                        "sheetId": answers_sheet_id,
                                        "dimension": "ROWS",
                                        "startIndex": effective_row_index,
                                        "endIndex": effective_row_index + 1
                                    }
                                }
                            }
                        ]
                    }
                    self.service.spreadsheets().batchUpdate(
                        spreadsheetId=self.spreadsheet_id,
                        body=delete_request
                    ).execute()

                    # Increase the counter so that future deletions use the adjusted index.
                    deleted_count += 1

        return cprocessed

    def get_sheet_id(self, sheet_name):
        """
        Retrieve the sheetId for a given sheet title.
        """
        spreadsheet = self.service.spreadsheets().get(spreadsheetId=self.spreadsheet_id).execute()
        for sheet in spreadsheet.get('sheets', []):
            props = sheet.get("properties", {})
            if props.get("title") == sheet_name:
                return props.get("sheetId")
        raise Exception(f"Sheet '{sheet_name}' not found.")

