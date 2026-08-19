import os
import json

class TwilioVoiceNotifier:
    """
    Handles Twilio Programmable Voice calls for confirmed critical behavioral deviations.
    Architecture MUST be: Uno Q -> MQTT -> Backend -> Twilio -> Operator Phone.
    Never put Twilio credentials on Uno Q.
    Uses environment variables: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER.
    If credentials are missing or call fails, safely falls back to MOCK MODE.
    """
    def __init__(self):
        self.account_sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        self.from_number = os.environ.get("TWILIO_FROM_NUMBER", "")
        
        self.is_configured = bool(self.account_sid and self.auth_token and self.from_number)
        
        if self.is_configured:
            try:
                from twilio.rest import Client
                self.client = Client(self.account_sid, self.auth_token)
                print("[TWILIO] Successfully initialized Twilio Voice Client.")
            except Exception as e:
                print(f"[TWILIO WARNING] Failed to initialize Twilio Client ({e}). Defaulting to MOCK MODE.")
                self.is_configured = False
                self.client = None
        else:
            self.client = None
            print("[TWILIO] Credentials not found in environment variables. Operating in SAFE MOCK MODE.")

    def trigger_voice_alert(self, to_phone, machine_name, machine_id, location, drift_score, top_features):
        """
        Triggers an automated voice call or logs mock voice alert.
        Message states 'critical behavioral deviation' without claiming an unvalidated specific fault.
        """
        message_text = (
            f"Alert. RetroFit Critical Behavioral Deviation detected for machine {machine_name}, ID {machine_id}, "
            f"located at {location}. Behavioral drift is {drift_score:.1f}. "
            f"Primary contributing signals: {', '.join(top_features[:2])}. Please inspect the machine immediately."
        )
        
        if self.is_configured and self.client and to_phone:
            try:
                twiml_url = f"http://twimlets.com/message?Message%5B0%5D={message_text.replace(' ', '%20')}"
                call = self.client.calls.create(
                    to=to_phone,
                    from_=self.from_number,
                    url=twiml_url
                )
                print(f"[TWILIO REAL CALL] Voice alert call placed to {to_phone}. Call SID: {call.sid}")
                return {'status': 'SUCCESS', 'mode': 'REAL_TWILIO', 'call_sid': call.sid, 'message': message_text}
            except Exception as e:
                print(f"[TWILIO ERROR] Call placement failed ({e}). Falling back to Mock Mode.")
                return {'status': 'FAILED_FALLBACK_MOCK', 'mode': 'MOCK', 'error': str(e), 'message': message_text}
        else:
            print(f"\n[TWILIO MOCK VOICE CALL ALERT]")
            print(f" -> Target Phone : {to_phone if to_phone else 'NOT_PROVIDED (Mock Default: +15550199)'}")
            print(f" -> Message Text : \"{message_text}\"")
            print(f" -> Mode         : SAFE MOCK MODE (No external API charges)\n")
            return {'status': 'SUCCESS', 'mode': 'MOCK', 'to': to_phone, 'message': message_text}
