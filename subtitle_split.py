import json
import re
import argparse
import sys
import time
import boto3
from botocore.exceptions import ClientError

#region GLOBAL VARIABLES
TOKENIZE_TEXT = ""
#endregion

#region Global language dictionary (code → language name)
LANGUAGES = {
    "sq": "Albanian",
    "ar": "Arabic",
    "hy": "Armenian",
    "bn": "Bengali",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English (United States)",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian / Farsi",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "ga": "Irish",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ku": "Kurdish",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "my": "Myanmar (Burmese)",
    "nb": "Norwegian Bokmål",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese"
}
#endregion

#region AWS Bedrock Config
MAX_RETRIES = 3
RETRY_THROTTLE_DELAY = 1.5  # seconds
default_model_region = "us-west-2"
access_key = "YOUR_AWS_ACCESS_KEY"
secret_key = "YOUR_AWS_SECRET_KEY"
default_model_id = "us.meta.llama3-3-70b-instruct-v1:0"
default_max_gen_len = 2048
default_temperature = 0.7
default_top_p = 0.9
llm_max_retry = 7
#endregion

#region Initialize LLM Client
llm_client = boto3.client(
    "bedrock-runtime",
    region_name=default_model_region,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)
#endregion

#region Core Process Function

def process(prompt, model_id, temperature=0.7, top_p=0.9, max_gen_len=2048, max_retry=MAX_RETRIES, verbose=True):
    """Send prompt to AWS Bedrock model and return generated text."""
    if not llm_client:
        raise ValueError("Bedrock access has not been authenticated.")
    if not prompt.strip():
        raise ValueError("Prompt must be provided.")
    if not model_id.strip():
        raise ValueError("Model ID must be provided.")

    # Format prompt for chat-style LLM
    prompt_formatted = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        "You are a helpful assistant<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    payload = {
        "prompt": prompt_formatted,
        "max_gen_len": max_gen_len,
        "temperature": temperature,
        "top_p": top_p,
    }

    for i in range(max_retry):
        try:
            response = llm_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                body=json.dumps(payload),
            )

            response_body = json.loads(response["body"].read())
            generated_text = response_body.get("generation")

            if generated_text:
                return generated_text.strip()

        except ClientError as e:
            if e.response['Error']['Code'] == "ThrottlingException":
                if verbose:
                    print(f"Throttling detected, retrying in {RETRY_THROTTLE_DELAY**i:.1f} seconds...")
                time.sleep(RETRY_THROTTLE_DELAY**i)
                continue
            else:
                if verbose:
                    print(f"ClientError: {e}")
        except Exception as e:
            if verbose:
                print(f"Unexpected error: {e}")
            break

        time.sleep(1)

    if verbose:
        print(f"Failed after {max_retry} retries.")
    return None

#endregion

#region Helper for Extracting JSON Attribute

def clean_escaped_quotes(json_string):
        # Replace escaped quotes (\") with unescaped quotes (")
        cleaned_string = re.sub(r'\\"', '"', json_string)
        return cleaned_string

def fix_bad_quote_escapes(json_input_string):
        
        json_string = json_input_string
        if json_string:

            # Step 1: Remove all multi-spaces to a single space and split each attribute onto a new line
            json_string_formatted = ' '.join(json_string.split())
            json_string_formatted = re.sub(r"(\{|\[)\s*(\"|\')", r"\1\n\2", json_string_formatted)
            json_string_formatted = re.sub(r"(\"|\'|True|False|\d+(\.\d+))\s*(\}|\])", r'\1\n\3', json_string_formatted)

            # Loop through and check each line
            lines = json_string_formatted.splitlines()
            checked_output = []
        
            for line in lines:
                line = line.strip()
                if line.startswith('{') or line.startswith('[') or line.endswith('}') or line.endswith(']'):
                    # Skip empty lines and braces/brackets on their own lines
                    checked_output.append(line)
                    continue
            
                 # Split line into attribute and value
                 # First char will determine whether the attribute is wrapped by ' or "
                ## TODO Check to see if there is a colon and handle if not e.g. [{ }]
                line_attribute, line_value = line.split(':', 1)
                line_attribute = line_attribute.strip()
                # remove single and quotes from the attribute
                line_attribute = re.sub(r'"', r'', line_attribute)
                line_attribute = re.sub(r"'", r"", line_attribute)
                line_attribute = line_attribute.strip()
            
                #Value checks
                has_comma = ""
                line_value = line_value.strip()
                value_wrapper_char = line_value[0] if line_value else None
                # Check if the value is a string (starts and ends with quotes)
                if value_wrapper_char in ('"', "'"):
                    # Get the last char from the value
                    last_char = line_value[-1]
                    if last_char == ",":
                        has_comman = ","
                        line_value = line_value[:-1].strip()

                    # Remove the first char 
                    line_value = line_value[1:]

                    # Remove value_wrapper_char from the head and tail of the string
                    if line_value.endswith(value_wrapper_char):
                        # remove the last char from start and end so we have just the string value
                        line_value = line_value[:-1]

                     ## We are now ready to check for encoded quotes
                    line_value = re.sub(r'\\"', '"', line_value)
                    line_value = re.sub(r"\\'", "'", line_value)

                    # escape single and double quotes
                    line_value = re.sub(r'"', r'\"', line_value)
                    #line_value = re.sub(r"'", r"\'", line_value)
                
                    # Add the wrapper char back to the head and tail of the string
                    #line_value = value_wrapper_char + line_value + value_wrapper_char


                    # Append the checked attribute and value
                    checked_output.append(f"\"{line_attribute}\":\"{line_value}{has_comma}\"")
                else:
                    ## TODO - detect if there is a string tail, if not, it is a number or boolean

                    checked_output.append(line)
                    continue
            validated_json = "".join(checked_output)
            # Return the formatted and checked JSON-like string
            return validated_json
        else:
            return json_string

def clean_and_balance_json(json_string):
        # Step 1: Count '{' and '}' in the extracted content
        open_braces = json_string.count('{')
        close_braces = json_string.count('}')
    
        if open_braces==close_braces:
            # Braces are already balanced
            return json_string

        # Step 2: Balance the braces if there's a mismatch
        if open_braces > close_braces:
            print("More opening braces than closing braces; this JSON might be truncated.")
            return None  # Can't fix if there are missing closing braces
        
        elif close_braces > open_braces:
            # Remove excess closing braces from the end
            excess_closing = close_braces - open_braces
            potential_json = potential_json.rstrip('}' * excess_closing)
    
        # Step 3: Attempt to parse the cleaned JSON string
        try:
            parsed_json = json.loads(potential_json)
            return parsed_json  # Return the parsed JSON if successful
        except json.JSONDecodeError as e:
            print("JSON is still invalid after cleaning:", e)
            return None

def attempt_to_fix_invalid_json(json_string):
        work_json_string = json_string
        work_json_string = clean_escaped_quotes(work_json_string)
        valid_json = is_valid_json(work_json_string)
        if not valid_json:
            ## Try removing junk from start and end
            work_json_string = clean_json_string(work_json_string)

            valid_json = is_valid_json(work_json_string)
            if not valid_json:
                ## Try to fix quote escapes
                work_json_string = fix_bad_quote_escapes(work_json_string)

                valid_json = is_valid_json(work_json_string)
                if not valid_json:
                    ## Check the number of { } are balanced
                    work_json_string = clean_and_balance_json(work_json_string)

        return valid_json, work_json_string, json_string

def extract_attribute_value(data_str, attribute):

        try:
            # Step 1: Check if the input JSON data is valid
            # Use regex to find the JSON object within the string
            valid_json = is_valid_json(data_str)
            if not valid_json:
                # Invalid, so attempt to fix the JSON
                valid_json, validated_json_string, json_string = attempt_to_fix_invalid_json(data_str)
                if valid_json:
                    data_str = validated_json_string
                else:
                    return None

            # Chekc if the JSON if valid
            if valid_json:
                json_match = re.search(r"{.*}", data_str, re.DOTALL)
                if json_match:
                    try:
                        # Extract the JSON string and load it as a dictionary
                        json_data = json.loads(json_match.group(0))

                        # Check if the "tape" attribute exists and return its value
                        if attribute in json_data:
                            return json_data[attribute].strip()
                    except json.JSONDecodeError:
                        print("Invalid JSON data found.")
                else:
                    print("No JSON data found.")
            else:
                print("Invalid JSON data found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return None

#endregion

#region Post-Editing Subttile Split to apply in JS script

def post_edit_subtitle_split(llm_output):
    """
    Post-edits LLM output subtitle splits:
    - If single line: Returns text as is (no pipes).
    - If multiple lines (due to LSBRLS or LLM split): 
      - First segment: "Text|"
      - Middle segments: "|Text|"
      - Last segment: "|Text"
    """
    if not llm_output:
        return ""

    segments = [seg.strip() for seg in llm_output.split("|") if seg.strip()]
    
    # If there is only one segment (short text, no LSBRLS), return it without pipes
    if len(segments) == 1:
        return segments[0]

    # Build the post-edited string for multi-segment subtitles
    post_edited_segments = []
    for i, seg in enumerate(segments):
        if i == 0:
            # First segment ends with pipe
            post_edited_segments.append(f"{seg}|")
        elif i == len(segments) - 1:
            # Last segment starts with pipe
            post_edited_segments.append(f"|{seg}")
        else:
            # Middle segments start and end with pipe
            post_edited_segments.append(f"|{seg}|")

    # Join with line breaks
    return "\n".join(post_edited_segments)

#endregion

#region Main Subtitle Split Function

def clean_json_string(json_string):
        if not json_string == None:
            try:
                # Use regex to capture content between the first '{' and the last '}'
                match = re.search(r'\{.*\}', json_string, re.DOTALL)
                if match:
                    # Return the cleaned JSON content
                    return match.group(0)
                else:
                    # Return None if no valid JSON structure is found
                    return None
            except Exception as e:
                return None
        else:
            return None

def is_valid_json(json_string):
        try:
            if json_string == None or len(json_string.strip()) == 0:
                return False
            
            # Try to load the JSON
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

def get_language(code: str) -> str:
    """
    Retrieve the language name by its code.
    Returns 'Unknown language code' if not found.
    """
    return LANGUAGES.get(code, "Unknown language code")

#region Applying Validation with Thai Segments with LLM Outputs

def clean_subtitle_tags_lst(token_list):
    """
    Removes SSA/ASS style formatting characters ({, }, \\, an1-9) 
    and HTML-style tags (<i>, </b>, etc.) from a list of tokens.
    """
    cleaned_list = []
    
    # Compile regex patterns for better performance
    # Pattern for HTML tags (e.g., <i>, <font color="...">)
    html_pattern = re.compile(r'<[^>]+>')
    
    # Pattern for alignment tags an1 through an9 (case insensitive)
    align_pattern = re.compile(r'^an[1-9]$', re.IGNORECASE)
    
    # Set of exact symbols to remove
    symbols_to_remove = {'{', '}', '\\'}

    for token in token_list:
        if not token:
            continue
            
        t = token.strip()
        
        # 1. Remove specific symbols ({, }, \)
        if t in symbols_to_remove:
            continue
            
        # 2. Remove alignment tags (an1 - an9)
        if align_pattern.match(t):
            continue
            
        # 3. Remove HTML tags
        if html_pattern.match(t):
            continue
            
        # If valid, add to list
        cleaned_list.append(token)
        
    return cleaned_list

def robust_thai_splitter(original_text, llm_output, max_len):
    """
    Splits Thai text based on LLM suggestions but strictly enforces:
    1. No hallucinations (rebuilds using ONLY original text).
    2. No broken words (uses PyThaiNLP tokens).
    3. MAX_LEN limit (forces a split if the segment gets too long).
    4. Conjunction Preference (tries to split BEFORE words like แต่, และ).
    """

    '''
    if not word_tokenize:
        return llm_output  # Fallback if library missing
    '''

    # A. Tokenize Original (Immutable Grid)
    # tokens = word_tokenize(original_text, engine="newmm")

    #Ensure original text is sanitized before tokenizing
    # clean_original = sanitize_input(original_text)      
    # tokens = word_tokenize(clean_original, engine="newmm")

    #Tokenized Text directly from the input argument
    tokens = TOKENIZE_TEXT.split(" ")
    # tokens = clean_subtitle_tags_lst(tokens)


    # print("___Thai Tokens____:\t", tokens, "\n")
    # print("__llm_output___:\t", llm_output, "\n")
    
    # B. Map LLM Intent to Tokens
    split_mask = [False] * len(tokens)
    clean_llm = llm_output.replace(" ", "").replace("\n", "").replace("|", "#") # mark pipes temporarily
    
    # We reconstruct the 'clean' string without pipes to find positions
    # But checking strictly against LLM characters is hard if LLM hallucinated.
    # STRATEGY: We look for the existence of the 'pipe' relative to token text.
    
    # Simple mapping: normalized LLM string with pipes
    # llm_normalized = llm_output.replace(" ", "").replace("\n", "")
    llm_normalized = sanitize_input(llm_output).replace(" ", "").replace("\n", "")
    
    current_search_idx = 0
    for i, token in enumerate(tokens):
        clean_token = token.replace(" ", "").replace("\n", "")
        if not clean_token: continue

        match_idx = llm_normalized.find(clean_token, current_search_idx)
        
        if match_idx != -1:
            end_of_token = match_idx + len(clean_token)
            
            # Check if the next char in LLM output is a pipe '|'
            if end_of_token < len(llm_normalized) and llm_normalized[end_of_token] == "|":
                split_mask[i] = True
                current_search_idx = end_of_token + 1
            else:
                current_search_idx = end_of_token

    # C. Construct Lines & Enforce Rules
    final_lines = []
    current_line_tokens = []
    current_line_len = 0
    
    # "Prefer splitting BEFORE these, so we check the PREVIOUS token.
    conjunctions = {"แต่", "และ", "หรือ", "ที่", "ซึ่ง", "ว่า", "เพื่อ", "โดย", "ถ้า", "หาก", "เพราะ", "จน", "เมื่อ"}

    for i, token in enumerate(tokens):
        token_len = len(token)
        
        # 1. HARD LIMIT CHECK
        is_overflow = (current_line_len + token_len) > max_len
        
        # 2. LLM REQUEST CHECK (Did LLM put a pipe after the previous token?)
        llm_requested_split = (i > 0) and split_mask[i-1]
        
        # 3. CONJUNCTION CHECK (Soft limit)
        # If current token is a conjunction, and line is already > 50% full, force split before it.
        is_conjunction = token in conjunctions
        force_conjunction_split = is_conjunction and (current_line_len > (max_len * 0.5))

        # DECISION: Split if Overflow OR (LLM asked AND not too short) OR Conjunction logic
        should_split = False
        
        if is_overflow:
            should_split = True
        elif llm_requested_split:
            should_split = True
        elif force_conjunction_split:
            should_split = True

        if current_line_tokens and should_split:
            # Commit current line
            final_lines.append("".join(current_line_tokens))
            # Start new line with current token
            current_line_tokens = [token]
            current_line_len = token_len
        else:
            # Append to current line
            current_line_tokens.append(token)
            current_line_len += token_len

    # Flush buffer
    if current_line_tokens:
        final_lines.append("".join(current_line_tokens))

    return "|".join(final_lines)

def split_subtitle_by_LLM(subtitle_lang_code, subtitle_sentence, tokenize_subtitle_text, max_char_len):
    """
    Splits subtitle text.
    - Routes Thai to 'robust_thai_splitter'.
    - Routes others to 'optimize_segments'.
    """
    
    global TOKENIZE_TEXT
    # TOKENIZE_TEXT = tokenize_subtitle_text

    # 1. Handle forced pre-existing line breaks
    tokenize_subtitle_text_lst = tokenize_subtitle_text.split("LSBRLS")
    raw_segments = subtitle_sentence.split("LSBRLS")
    final_segments = []
    
    is_thai = (subtitle_lang_code.lower() in ["th", "tha"])

    for idx, segment in enumerate(raw_segments):
        segment = segment.strip()
        if not segment:
            continue
        
        TOKENIZE_TEXT = tokenize_subtitle_text_lst[idx]
        clean_segment = clean_subtitle_tags(segment)

        # 2. Check length
        if len(clean_segment) <= max_char_len:
            final_segments.append(segment)
        else:
            # 3. Query LLM
            llm_split_result = _query_llm_for_split(subtitle_lang_code, segment, max_char_len)
            
            if llm_split_result:
                '''# if is_thai and word_tokenize:'''
                if is_thai:
                    # THAI PATH:
                    # Step A: Robust Validation (Fixes hallucinations & enforces PyThaiNLP boundaries)
                    validated_split = robust_thai_splitter(segment, llm_split_result, max_char_len)

                    # Step B: Optimization (Merges segments back if they fit, to avoid over-splitting)
                    # We pass the 'validated_split' into the optimizer.
                    optimized_thai_result = optimize_segments(validated_split, segment, max_char_len)

                    final_segments.extend(optimized_thai_result.split("|"))
                else:
                    # OTHERS: Use Optimization (Merges back if fit)
                    optimized_result = optimize_segments(llm_split_result, segment, max_char_len)
                    final_segments.extend(optimized_result.split("|"))
            else:
                # Fallback
                final_segments.append(segment)

    # 4. Combine
    combined_text = "|".join(final_segments)

    # 5. Final Post-Edit
    return post_edit_subtitle_split(combined_text)

def optimize_segments(split_text, original_text, max_char_len):
    """
    Merges adjacent segments if they fit within max_char_len.
    (Used mainly for non-Thai languages)
    """
    if not split_text: return ""
    
    segments = [s.strip() for s in split_text.split("|") if s.strip()]
    if len(segments) < 2: return split_text.strip()

    aligned_data = []
    current_pos = 0
    
    for i, seg in enumerate(segments):
        start_index = original_text.find(seg, current_pos)
        if start_index == -1:
            sep = " " if i > 0 else ""
            aligned_data.append({"text": seg, "sep": sep})
            continue

        if i > 0:
            separator = original_text[current_pos:start_index]
        else:
            separator = ""

        aligned_data.append({"text": seg, "sep": separator})
        current_pos = start_index + len(seg)

    result_segments = []
    curr_buffer = aligned_data[0]['text']
    
    for i in range(1, len(aligned_data)):
        next_item = aligned_data[i]
        next_text = next_item['text']
        separator = next_item['sep']
        candidate = curr_buffer + separator + next_text
        
        if len(clean_subtitle_tags(candidate)) <= max_char_len:
            curr_buffer = candidate
        else:
            result_segments.append(curr_buffer)
            curr_buffer = next_text

    result_segments.append(curr_buffer)
    return "|".join(result_segments)

def clean_subtitle_tags(text):
    if not text: return ""
    text = re.sub(r'</?i>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\{\\an[1-9]\}', '', text)
    return text.strip()

def sanitize_input(text):
    """
    Fixes Python string escaping issues where '\\a' becomes the Bell character '\x07'.
    Converts '\x07' back to literal '\\a'.
    """
    if not text: return ""
    # Replace ASCII Bell (\x07) with literal \a
    if '\x07' in text:
        text = text.replace('\x07', '\\a')
    return text


#region Combination with D_Approach: Pre-Segmented Text to LLM

def prepare_segmented_input(text):
    """
    Tokenizes Thai text and joins with '~' to create strict boundaries.
    """
    
    # Sanitize input (fix \x07 issue first)
    text = text.replace('\x07', '\\a')
    
    # Tokenize
    # tokens = word_tokenize(text, engine="newmm")
    tokens = TOKENIZE_TEXT.split(" ")
    
    # Join with ~
    # We must be careful not to double-mark existing spaces
    # If token is " ", we keep it as " ". 
    # But ~ is inserted BETWEEN every token.
    segmented_text = "~".join(tokens)
    
    return segmented_text

def clean_tilde_artifacts(text):
    """
    Cleaner: Removes leftover tildes from LLM output.
    Handles messy cases: 'Word~|Word' -> 'Word|Word', 'Word~Word' -> 'WordWord'
    """
    if not text: return ""
    
    # 1. Fix mixed boundaries (LLM combined ~ and |)
    text = text.replace("~|", "|")
    text = text.replace("|~", "|")
    
    # 2. Remove remaining tildes (implies joining words)
    text = text.replace("~", "")
    
    return text

def _query_llm_for_split(subtitle_lang_code, subtitle_sentence, max_char_len):
    
    # CONFIGURATION: SET THIS TO TRUE TO APPLY WITH METHOD A FOR SENDING PRE-SEGMENT TXT TO LLM
    
    # Set to False: Method A (pre-segmentation) degrades performance compared to natural raw text pass to LLM.
    use_method_a_segmentation = False  

    current_len = len(subtitle_sentence)
    subtitle_language = get_language(subtitle_lang_code)
    attempt_count = 0
    not_fit_split_lst = []

    # Prepare Data based on Method
    if use_method_a_segmentation and subtitle_lang_code.lower() in ["th", "tha"]:
        input_text = prepare_segmented_input(subtitle_sentence)        
    else:
        input_text = subtitle_sentence        

    safe_sentence = json.dumps(input_text, ensure_ascii=False)

    # PROMPT 1: GENERRAL PROMPT
    prompt_generic = f"""
ROLE: You are a professional subtitle editor for the {subtitle_language} language.

TASK:
Split the provided subtitle sentence so that it:
- Respects the maximum character limit per line: {max_char_len}
- Breaks at natural linguistic and semantic boundaries (clause-based splitting appropriate for {subtitle_language}).
- Maintains the original meaning, tone, and nuance.
- Ensures smooth readability for viewers.

STRICT HARD LIMITS (CRITICAL):
1. MAX CHARACTER LENGTH: {max_char_len} characters per segment.
2. PRIORITY RULE: The Character Length Limit is STRICTER than grammatical perfection. 
   - If a natural linguistic split creates a segment longer than {max_char_len}, you MUST break the segment earlier to satisfy the limit.
   - It is better to have a slightly imperfect split than to exceed the character limit.

UNIVERSAL LINGUISTIC RULES (Apply to ALL languages):
1. Numeric Cohesion (Numbers & Units):
   - NEVER separate a number from its unit or following quantifier.
   - Keep "Number + Space + Word" together.
   - Example: Keep "10 minutes", "100 km", "5 items" together.

ABSOLUTE NON-NEGOTIABLE:
- No words, punctuation, or spaces may be dropped, changed, or altered in any way in the target text. 
- The only allowed operation is to add a "|" marker between segments.
- The character length of each segment (i.e., between the "|" markers) MUST be less than or equal to {max_char_len}.
- Do not create unnecessary very short segments: if two adjacent segments combined are ≤ {max_char_len} characters, they must remain a single segment.

GUIDELINES (apply according to {subtitle_language} grammar rules):
1. Identify natural clause boundaries first, then place split markers at the point closest to the left of the {max_char_len} limit.
2. Avoid placing a split immediately after short function words or equivalents.
3. Do NOT split immediately after subordinating conjunctions.
4. Do NOT split within fixed expressions or idioms.
5. Avoid placing a split immediately before punctuation; instead, split after punctuation if it marks a natural pause.
6. Prefer longer segments, aiming to approach {max_char_len} characters.
7. Avoid short segments with fewer than half of {max_char_len} characters unless unavoidable.
8. Keep related words or characters together.
9. Split where a natural pause occurs.
10. Use a pipe character (|) with no spaces before or after it to mark each split point.
11. Do not alter, add, or remove words or characters from the original text.

OUTPUT FORMAT:
Return only the following JSON format with no commentary or extra text:
{{
"split_subtitle": "<subtitle with | markers at split points>"
}}

INPUT DATA:
{{
"text": {safe_sentence},
"current_length": {current_len},
"max_allowed_length": {max_char_len},
"instruction": "Split 'text' so no part exceeds 'max_allowed_length'."
}}
"""        

    # PROMPT 2: METHOD A (PRE-SEGMENTED THAI TEXT TO LLM) PROMPT FOR THAI LANG
    prompt_method_a = f"""
ROLE: You are a professional subtitle editor for the Thai language.

TASK:
The input text is pre-segmented into word tokens using the tilde character "~" (e.g., "คำ1~คำ2~คำ3").
Your task is to process these tokens to form subtitle lines that:
- Respect the maximum character limit per line: {max_char_len}
- Break at natural linguistic and semantic boundaries.
- Maintain the original meaning, tone, and nuance.

OPERATIONAL LOGIC (THE "~" RULE):
You must make a decision for EVERY "~" in the input:
1. **JOIN (Default):** If no line break is needed, REMOVE the "~" completely to join the words (e.g., "บ้าน~ฉัน" -> "บ้านฉัน").
2. **SPLIT (Line Break):** If a line break is needed, REPLACE the "~" with a "|" (e.g., "ไป~แล้ว" -> "ไป|แล้ว").

STRICT HARD LIMITS (CRITICAL):
1. MAX CHARACTER LENGTH: {max_char_len} characters per segment (text between "|").
2. PRIORITY RULE: The Character Length Limit is STRICTER than grammatical perfection. 
   - If joining tokens creates a segment > {max_char_len}, you MUST replace a "~" with "|" earlier to satisfy the limit.

ABSOLUTE NON-NEGOTIABLE:
- Do NOT output any "~" characters in the final result. All "~" must be either removed or changed to "|".
- No words, punctuation, or spaces may be dropped, changed, or altered.
- Do not create unnecessary very short segments.

GUIDELINES (Thai Grammar & Readability):
1. Identify natural clause boundaries first.
2. **Start-With Rule (Conjunctions):** 
   - If a "~" appears immediately BEFORE a conjunction, REPLACE it with "|".
   - **Target Words:** แต่ (but), และ (and), หรือ (or), ที่ (that), ซึ่ง (which), ว่า (that), กับ (with), โดย (by), เพื่อ (for), ถ้า (if), หาก (if).
   - *Example:* "อยากหย่า~แต่" -> "อยากหย่า|แต่" (Correct) vs "อยากหย่าแต่" (Avoid if long).
3. **Keep Together:** 
   - REMOVE the "~" between a Prefix and its root (e.g., "การ~เดินทาง" -> "การเดินทาง").
   - REMOVE the "~" between Number and Classifier (e.g., "1~นาที" -> "1นาที").
4. Avoid splitting immediately before punctuation (.,?!).
5. Prefer longer segments, aiming to approach {max_char_len}.

CRITICAL SECURITY RULES (ZERO TOLERANCE):
1. IMMUTABLE TEXT: The text characters (excluding "~") must match the input exactly.
2. NO DUPLICATION: Do NOT repeat words (e.g., Input "รอ" -> Output "รอ". NEVER "rอ รอ").
3. NO REPAIR: Do NOT fix typos. Do NOT add spaces where there were none.
4. CLEANUP: Ensure NO tildes "~" remain in the output.

OUTPUT FORMAT:
Return only the following JSON format with no commentary:
{{
"split_subtitle": "<cleaned_text_with_pipes>"
}}

INPUT DATA:
{{
"text": {safe_sentence},
"current_length": {current_len},
"max_allowed_length": {max_char_len},
"instruction": "Process the '~' tokens. Replace '~' with '|' for breaks, or remove '~' to join. STRICTLY follow {max_char_len}."
}}
"""

    # PROMPT 3: METHOD B (ROBUST RAW WITHOUT PRE-SEGMENTATION FOR THAI LANG - DEFAULT)
    prompt_method_b = f"""
ROLE: You are a professional subtitle editor for the Thai language.
 
TASK:
Split the provided subtitle sentence so that it:
- Respects the maximum character limit per line: {max_char_len}
- Breaks at natural linguistic and semantic boundaries (clause-based splitting appropriate for Thai).
- Maintains the original meaning, tone, and nuance.
- Ensures smooth readability for viewers.
STRICT HARD LIMITS (CRITICAL):
1. MAX CHARACTER LENGTH: {max_char_len} characters per segment.
2. PRIORITY RULE: The Character Length Limit is STRICTER than grammatical perfection. 
   - If a natural linguistic split creates a segment longer than {max_char_len}, you MUST break the segment earlier to satisfy the limit.
   - It is better to have a slightly imperfect split than to exceed the character limit.
 
ABSOLUTE NON-NEGOTIABLE:
- No words, punctuation, or spaces may be dropped, changed, or altered in any way in the target text. 
- The only allowed operation is to add a "|" marker between segments.
- The character length of each segment (i.e., between the "|" markers) MUST be less than or equal to {max_char_len}.
- Do not create unnecessary very short segments: if two adjacent segments combined are ≤ {max_char_len} characters, they must remain a single segment.
 
GUIDELINES (apply according to Thai grammar rules):
1. Identify natural clause boundaries first, then place split markers at the point closest to the left of the {max_char_len} limit.
2. Thai does not typically use spaces between words, so split decisions must focus on semantic units and clause structure.
3. Avoid splitting within a single Thai word or a tightly connected compound word (e.g., proper nouns, technical terms, fixed expressions).
4. Prefer splitting after natural pause markers, conjunctions (เชื่อม), particles (คำช่วย), or at the end of a clause or phrase.
5. Avoid splitting immediately before or within a classifier/measure word (ลักษณนาม) and the noun it modifies.
6. Do NOT split immediately after subordinating conjunctions or prepositions (e.g., "แต่", "ที่", "ซึ่ง", "ว่า", "กับ", "โดย", "เพื่อ" etc., when they connect clauses/phrases).
7. Avoid placing a split immediately before punctuation (such as "," "?" "!") unless the punctuation marks a clear clause end and helps satisfy the character limit. Instead, split after punctuation.
8. Prefer longer segments, aiming to approach {max_char_len} characters.
9. Avoid short segments with fewer than half of {max_char_len} characters unless unavoidable.
10. Keep related words or characters together.
11. Split where a natural pause occurs.
12. Use a pipe character (|) with no spaces before or after it to mark each split point.
13. Do not alter, add, or remove words or characters from the original text. 

CRITICAL SECURITY RULES (ZERO TOLERANCE):
1. IMMUTABLE TEXT: The output text (ignoring "|") MUST match the input text exactly, character-for-character.
2. NO DUPLICATION: Do NOT repeat words. (Example: Input "รอ" -> Output "รอ". NEVER "rอ รอ").
3. NO REPAIR: Do NOT fix typos, do NOT change spaces, do NOT add English characters.
4. STRICT CHECK: If the input is "A B C", the output must be "A|B|C" or "A B|C". NOT "A b C".

CRITICAL RULE: LEADING CONJUNCTIONS (The "Start-With" Rule)
You must strictly follow this rule for conjunctions and prepositions:
- NEVER end a segment with a connecting word (e.g., "แต่|").
- ALWAYS place the split marker BEFORE the connecting word so it starts the new line.
- TARGET WORDS: แต่ (but), และ (and), หรือ (or), ที่ (that/at), ซึ่ง (which), ว่า (that), กับ (with), โดย (by), เพื่อ (for), เพราะ (because), ถ้า (if), หาก (if).
 
Examples:
- CORRECT: "บ้านเธอเลยอยากหย่า|แต่เธอกับสามีไม่มีทรัพย์สิน" (New line starts with แต่)
- WRONG:   "บ้านเธอเลยอยากหย่า แต่|เธอกับสามีไม่มีทรัพย์สิน" (Line ends with แต่)

OUTPUT FORMAT:
Return only the following JSON format with no commentary or extra text:
{{
"split_subtitle": "<subtitle with | markers at split points>"
}}
 
INPUT DATA:
{{
"text": {safe_sentence},
"current_length": {current_len},
"max_allowed_length": {max_char_len},
"instruction": "Split 'text' so no part exceeds 'max_allowed_length'. CRITICAL: If splitting near conjuction words like 'แต่', 'ที่', 'ว่า', insert '|' BEFORE them."
}}
"""        
    
    while attempt_count < llm_max_retry:
        attempt_count += 1
        
        # Select Prompt
        if subtitle_lang_code.lower() in ["th", "tha"]:
            final_prompt = prompt_method_a if use_method_a_segmentation else prompt_method_b
        else:
            final_prompt = prompt_generic

        try:
            response = process(
                final_prompt, default_model_id, default_temperature, 
                default_top_p, default_max_gen_len, llm_max_retry
            )
            if not response: continue

            cleaned_response = clean_json_string(response)
            if not is_valid_json(cleaned_response): continue

            split_text = extract_attribute_value(cleaned_response, "split_subtitle")
            
            if split_text:
                # SPECIAL CLEANING FOR METHOD A
                if use_method_a_segmentation and subtitle_lang_code.lower() in ["th", "tha"]:
                    split_text = clean_tilde_artifacts(split_text)

                # -----------------
                # Validation
                # -----------------

                if subtitle_lang_code.lower() in ["th", "tha"]:
                    # For Thai, we trust the robust_thai_splitter to fix lengths, 
                    # so we just return the LLM's "Intent" immediately.
                    return split_text.strip() # Trust Robust Splitter downstream
                
                # For strict length check
                if all(len(seg) <= max_char_len for seg in split_text.split("|")):
                    return split_text.strip()
                else:
                    not_fit_split_lst.append(split_text.strip())

        except Exception:
            pass

    if not_fit_split_lst:
        best_candidate = min(not_fit_split_lst, key=lambda s: max(len(seg) for seg in s.split("|")))
        return best_candidate

    return None

#endregion


#endregion


#region Main Function to Run

def validate_args(args):
    if not args.subtitle_language:
        raise ValueError(f"subtitle_language is required.")

    if not args.subtitle_sentence:
        raise ValueError(f"subtitle_sentence is required.")

    if not args.max_char_len:
        raise ValueError(f"max_char_len is required.")
    
    # All validations passed
    return 

def main():
    parser = argparse.ArgumentParser("Tool for splitting subtitle text.")
    parser.add_argument('--subtitle_language', required=False, help="Subtitle language code (ISO 639-1, e.g., EN, ZH, PT, ES, etc). Used for language-specific splitting rules.", default="th")
    parser.add_argument('--subtitle_sentence', required=False, type=str, help="Subtitle text to split according to linguistic boundaries and max character limit.", default='เธอบอกว่า\" แค่ เพราะLSBRLSแค่ เพราะฉันแจ้งความว่าเขาหายไปไม่ได้แปลว่าฉันอยากได้เขาคืน\"ใครไปบาร์')    
    parser.add_argument('--tokenize_subtitle_text', required=False, type=str, help="Tokenized Subttile Text.", default='เธอ บอก ว่า \ " แค่ เพราะ LSBRLS แค่ เพราะ ฉัน แจ้งความ ว่า เขา หาย ไป ไม่ ได้ แปล ว่า ฉัน อยาก ได้ เขา คืน \ " ใคร ไป บาร์')    
    parser.add_argument('--max_char_len', required=False, type=int, help="Maximum number of characters allowed per subtitle line (including spaces).", default=37)

    args = parser.parse_args()

    try:
        validate_args(args)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

    subtitle_language = args.subtitle_language
    subtitle_sentence = args.subtitle_sentence
    tokenize_subtitle_text = args.tokenize_subtitle_text
    max_char_len = args.max_char_len
    subtitle_split_out = split_subtitle_by_LLM(subtitle_language, subtitle_sentence, tokenize_subtitle_text, max_char_len)
    # print("\nSubtitle Splits by LLM:\n", subtitle_split_out, "\n")
    print(subtitle_split_out)
    
#endregion

if __name__ == "__main__":
    main()