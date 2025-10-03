import streamlit as st
import pandas as pd
import requests  # noqa: F401
import json  # noqa: F401
import time
import os
from datetime import datetime
import zipfile
from io import BytesIO
import asyncio
import aiohttp
from typing import Optional, Dict, Any
import hashlib  # noqa: F401
import uuid
import contextlib

# Configuration - Updated with correct limits
DEFAULT_RPM_LIMIT = 20  # OpenRouter allows 20 RPM for free models
DEFAULT_DAILY_LIMIT = 50   # Free users: 50/day, Users with $10+ credits: 1000/day
API_BASE_URL = "https://openrouter.ai/api/v1"
OUTPUT_DIR_BASE = "outputs"

# Updated free models list with newer models
FREE_MODELS = {
    "deepseek/deepseek-r1-distill-llama-70b:free": {
        "name": "DeepSeek R1 Distill 70B",
        "context": 32000,
        "description": "Reasoning model with chain-of-thought capabilities",
        "recommended_temp": 0.1
    },
    "qwen/qwen-2.5-coder-32b-instruct:free": {
        "name": "Qwen 2.5 Coder 32B", 
        "context": 32768,
        "description": "Advanced coding and structured extraction specialist",
        "recommended_temp": 0.2
    },
    "qwen/qwen-2.5-72b-instruct:free": {
        "name": "Qwen 2.5 72B",
        "context": 32768, 
        "description": "High-performance general purpose model",
        "recommended_temp": 0.3
    },
    "x-ai/grok-4-fast:free": {
        "name": "Grok 4 Fast",
        "context": 2000000,
        "description": "Fast reasoning with 2M context",
        "recommended_temp": 0.2
    },
    "deepseek/deepseek-chat-v3.1:free": {
        "name": "DeepSeek V3.1",
        "context": 128000,
        "description": "Hybrid reasoning, best for legal extraction",
        "recommended_temp": 0.1
    },
    "qwen/qwen3-coder:free": {
        "name": "Qwen3 Coder",
        "context": 262144,
        "description": "Structured extraction specialist",
        "recommended_temp": 0.3
    },
    "anthropic/claude-3-haiku:beta": {
        "name": "Claude 3 Haiku",
        "context": 200000,
        "description": "Fast and efficient",
        "recommended_temp": 0.2
    },
    "meta-llama/llama-3.2-3b-instruct:free": {
        "name": "Llama 3.2 3B",
        "context": 131072,
        "description": "Compact model",
        "recommended_temp": 0.3
    }
}

def init_session_state():
    """Initialize session state variables"""
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if 'api_validated' not in st.session_state:
        st.session_state.api_validated = False
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    # Row selection & resume state
    if 'row_selection' not in st.session_state:
        st.session_state.row_selection = ""
    if 'last_row_selection' not in st.session_state:
        st.session_state.last_row_selection = ""
    if 'selected_row_indices' not in st.session_state:
        st.session_state.selected_row_indices = []
    if 'processing_cursor' not in st.session_state:
        st.session_state.processing_cursor = 0
    if 'auto_resume' not in st.session_state:
        st.session_state.auto_resume = False
    if 'chunk_size' not in st.session_state:
        st.session_state.chunk_size = 10
    if 'pending_indices' not in st.session_state:
        st.session_state.pending_indices = []
    if 'selection_snapshot' not in st.session_state:
        st.session_state.selection_snapshot = []
    if 'total_selected' not in st.session_state:
        st.session_state.total_selected = 0
    if 'pause_requested' not in st.session_state:
        st.session_state.pause_requested = False
    if 'request_timeout' not in st.session_state:
        st.session_state.request_timeout = 120
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []

class OpenRouterAPI:
    def __init__(self, api_key: str, rpm_limit: int = 20, daily_limit: int = 1000):  # Updated default RPM
        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self.daily_limit = daily_limit
        self.request_times = []
        self.daily_requests = 0
        self.last_reset = datetime.now().date()

    async def validate_api_key(self) -> tuple[bool, str]:
        """Validate API key by fetching models"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://streamlit-openrouter-batch.app",
            "X-Title": "OpenRouter Batch Processor"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{API_BASE_URL}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        return True, "API key validated successfully"
                    elif response.status == 401:
                        return False, "Invalid API key"
                    else:
                        return False, f"API validation failed: HTTP {response.status}"
        except Exception as e:
            return False, f"Connection error: {str(e)}"

    def check_rate_limits(self):
        """Check if we can make a request based on rate limits"""
        now = datetime.now()
        
        if now.date() > self.last_reset:
            self.daily_requests = 0
            self.last_reset = now.date()
        
        if self.daily_requests >= self.daily_limit:
            return False, "Daily limit reached"
        
        one_minute_ago = now.timestamp() - 60
        self.request_times = [t for t in self.request_times if t > one_minute_ago]
        
        if len(self.request_times) >= self.rpm_limit:
            return False, f"Rate limit exceeded ({self.rpm_limit} RPM)"
        
        return True, "OK"

    def wait_for_rate_limit(self):
        """Calculate how long to wait before next request"""
        if len(self.request_times) < self.rpm_limit:
            return 0
        
        oldest_time = min(self.request_times)
        wait_time = 60 - (time.time() - oldest_time)
        return max(0, wait_time)

    async def make_request(self, session: aiohttp.ClientSession, content: str,
                          preset: Optional[str] = None, model: Optional[str] = None,
                          system_prompt: str = "", preset_method: str = "field",
                          use_internal_rate_limit: bool = True, timeout_s: float = 120,
                          **kwargs) -> Dict:
        """
        Make API request with corrected preset support
        
        preset_method: 'field' (default) or 'direct_reference'
        - 'field': Uses {"preset": "preset-name", "model": "fallback-model"} in request body
        - 'direct_reference': Uses "@preset/preset-name" as the model parameter
        """
        if use_internal_rate_limit:
            can_proceed, reason = self.check_rate_limits()
            if not can_proceed:
                return {"error": reason, "content": content}

            wait_time = self.wait_for_rate_limit()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit-openrouter-batch.app",
            "X-Title": "OpenRouter Batch Processor"
        }

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        body = {"messages": messages}

        # Corrected preset handling
        if preset:
            preset_slug = preset
            # Ensure '@preset/' prefix
            if not preset_slug.startswith("@preset/"):
                preset_slug = f"@preset/{preset_slug}"

            if preset_method == "direct_reference":
                # Method 1: Direct model reference - preset AS the model
                body["model"] = preset_slug
            else:
                # Method 2: Preset field - rely on preset's model unless user explicitly picks one
                body["preset"] = preset_slug
                if model:
                    body["model"] = model
            
            # Parameters will be shallow-merged with preset configuration
            if kwargs:
                body.update(kwargs)
        elif model:
            body["model"] = model
            if kwargs:
                body.update(kwargs)
        else:
            return {"error": "No model or preset specified", "content": content}

        try:
            async with session.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=body,
                timeout=aiohttp.ClientTimeout(total=timeout_s)
            ) as response:
                # Track usage for visibility, regardless of external rate limiter
                self.request_times.append(time.time())
                self.daily_requests += 1
                
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response": result["choices"][0]["message"]["content"],
                        "model_used": result.get("model", model or f"@preset/{preset}" if preset else "unknown"),
                        "usage": result.get("usage", {}),
                        "content": content
                    }
                else:
                    error_text = await response.text()
                    return {
                        "error": f"HTTP {response.status}: {error_text}",
                        "content": content
                    }
        except asyncio.TimeoutError:
            return {"error": "Request timeout", "content": content}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}", "content": content}

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    import re
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return filename[:100]

def save_response_as_markdown(response_data: Dict, id_value: str, output_dir: str):
    """Save individual response as markdown file"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{sanitize_filename(str(id_value))}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# Response for ID: {id_value}\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        
        if response_data.get("success"):
            f.write(f"**Model:** {response_data.get('model_used', 'Unknown')}\n\n")
            f.write("## Response\n\n")
            f.write(response_data["response"])
            
            if response_data.get("usage"):
                f.write("\n\n## Usage Statistics\n\n")
                usage = response_data['usage']
                f.write(f"- Input tokens: {usage.get('prompt_tokens', 'N/A')}\n")
                f.write(f"- Output tokens: {usage.get('completion_tokens', 'N/A')}\n")
                f.write(f"- Total tokens: {usage.get('total_tokens', 'N/A')}\n")
        else:
            f.write("## Error\n\n")
            f.write(f"```\n{response_data.get('error', 'Unknown error')}\n```")
    
    return filepath

def create_zip_download(output_dir: str) -> bytes:
    """Create ZIP file of all markdown files"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, output_dir)
                    zip_file.write(file_path, arc_name)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def build_responses_text(results_df: pd.DataFrame, id_column: str) -> str:
    """Create a plain-text representation with only model responses."""
    if results_df.empty:
        return "No responses available.\n"

    sort_cols: list[str] = []
    if 'row_number' in results_df.columns:
        sort_cols.append('row_number')
    elif id_column in results_df.columns:
        sort_cols.append(id_column)

    if sort_cols:
        sorted_df = results_df.sort_values(by=sort_cols)
    else:
        sorted_df = results_df.copy()

    delimiter = "----- RESPONSE DELIMITER -----"
    entries: list[str] = []

    for _, row in sorted_df.iterrows():
        response = (row.get('response') or '').strip()
        if not response:
            continue
        identifier = row.get(id_column) or row.get('row_number') or 'row'
        entry_lines = [
            delimiter,
            f"ID: {identifier}",
            delimiter,
            response,
            delimiter
        ]
        entries.append("\n".join(entry_lines))

    if not entries:
        return "No responses available.\n"

    return "\n".join(entries) + "\n"


def append_debug_log(message: str):
    """Append a message to the session-scoped debug console."""
    log: list[str] = list(st.session_state.get('debug_log', []))
    log.append(message)
    # Keep the log from growing without bound
    if len(log) > 500:
        log = log[-500:]
    st.session_state.debug_log = log

async def process_batch(api_client: OpenRouterAPI, df: pd.DataFrame,
                       id_column: str, content_column: str,
                       preset: Optional[str], model: Optional[str],
                       system_prompt: str,
                       row_indices: list[int],
                       batch_params: Dict, preset_method: str = "field",
                       rpm_limit: int = DEFAULT_RPM_LIMIT, max_concurrency: int = 8,
                       request_timeout: float = 120,
                       progress_callback=None):
    """Process specific rows concurrently with RPM pacing and enhanced preset support"""
    output_dir = os.path.join(OUTPUT_DIR_BASE, f"session_{st.session_state.session_id}")

    # Preserve previously processed rows so we can resume across chunks
    accumulated_results = list(st.session_state.get('processing_results', []))
    result_index: Dict[tuple[int, Any], int] = {}
    id_key_name = id_column
    for idx, existing in enumerate(accumulated_results):
        row_no = existing.get('row_number')
        id_val = existing.get(id_key_name)
        if row_no is not None and id_val is not None:
            result_index[(row_no, id_val)] = idx
    results = []

    # Token bucket for RPM pacing
    token_queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, rpm_limit))
    # Pre-fill with initial tokens to allow immediate burst up to RPM
    for _ in range(max(1, rpm_limit)):
        token_queue.put_nowait(True)

    stop_refill = asyncio.Event()

    async def refill_tokens():
        interval = 60.0 / max(1, rpm_limit)
        try:
            while not stop_refill.is_set():
                await asyncio.sleep(interval)
                # Refill up to capacity
                if not token_queue.full():
                    token_queue.put_nowait(True)
        except asyncio.CancelledError:
            pass

    # Concurrency limiter
    concurrency_sem = asyncio.Semaphore(max(1, min(max_concurrency, rpm_limit)))

    total_in_chunk = len(row_indices)
    completed = 0

    async def handle_row(display_row_num: int, df_index: int, row: pd.Series):
        nonlocal completed, results
        id_value = row[id_column]
        content = str(row[content_column])

        # Pace by RPM (do not wait for prior responses)
        await token_queue.get()
        if progress_callback:
            progress_callback(completed, total_in_chunk, f"Queued row {display_row_num}: {str(id_value)}")
        append_debug_log(
            f"[{datetime.now().strftime('%H:%M:%S')}] QUEUED row {display_row_num} (ID={id_value})"
        )

        async with concurrency_sem:
            async with aiohttp.ClientSession() as session:
                result = await api_client.make_request(
                    session=session,
                    content=content,
                    preset=preset,
                    model=model,
                    system_prompt=system_prompt,
                    preset_method=preset_method,
                    use_internal_rate_limit=False,
                    timeout_s=request_timeout,
                    **batch_params
                )

        # Persist and surface
        markdown_path = save_response_as_markdown(result, id_value, output_dir)
        result_row = {
            'row_number': df_index + 1,
            id_column: id_value,
            content_column: content[:200] + "..." if len(content) > 200 else content,
            'response': result.get('response', ''),
            'error': result.get('error', ''),
            'success': result.get('success', False),
            'model_used': result.get('model_used', ''),
            'markdown_file': os.path.basename(markdown_path)
        }
        if result.get('usage'):
            usage = result['usage']
            result_row.update({
                'input_tokens': usage.get('prompt_tokens', 0),
                'output_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            })

        results.append(result_row)

        # Update accumulated results without dropping prior chunks
        key = (result_row.get('row_number'), result_row.get(id_key_name))
        if key in result_index:
            accumulated_results[result_index[key]] = result_row
        else:
            result_index[key] = len(accumulated_results)
            accumulated_results.append(result_row)

        st.session_state.processing_results = accumulated_results.copy()

        # Emit debug entry with truncated payload
        preview = result.get('response') if result.get('success') else result.get('error', '')
        preview = (preview or '').strip().replace('\n', ' ')
        if len(preview) > 180:
            preview = preview[:177] + '...'
        status = 'OK' if result.get('success') else 'ERROR'
        append_debug_log(
            f"[{datetime.now().strftime('%H:%M:%S')}] {status} row {display_row_num} (ID={id_value}) :: {preview or 'No content returned'}"
        )

        completed += 1
        if progress_callback:
            progress_callback(completed, total_in_chunk, f"Completed row {display_row_num}: {str(id_value)}")

    try:
        refill_task = asyncio.create_task(refill_tokens())

        # Build the ordered rows to process for this chunk
        rows = [(idx + 1, idx, df.iloc[idx]) for idx in row_indices]
        tasks = [asyncio.create_task(handle_row(display_num, frame_idx, row)) for display_num, frame_idx, row in rows]

        # Stream completions as they finish
        for coro in asyncio.as_completed(tasks):
            await coro
    finally:
        stop_refill.set()
        # Allow refill task to exit
        if 'refill_task' in locals():
            refill_task.cancel()
            with contextlib.suppress(Exception):
                await refill_task

    return results, output_dir

def validate_csv_data(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate uploaded CSV data"""
    if df.empty:
        return False, "CSV file is empty"
    
    if len(df.columns) < 2:
        return False, "CSV must have at least 2 columns (ID and content)"
    
    return True, "CSV validation passed"

def parse_row_selection(selection: str, total_rows: int) -> list[int]:
    """Parse a row selection string like '2-999,5,10-12' (1-based) into 0-based indices.
    Returns a sorted list of unique indices within bounds.
    """
    if not selection:
        return []
    indices = set()
    import re
    tokens = [t.strip() for t in selection.split(',') if t.strip()]
    for token in tokens:
        m = re.match(r"^(\d+)-(\d+)$", token)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            if start > end:
                start, end = end, start
            # clamp to bounds and convert to 0-based
            start0 = max(1, start)
            end0 = min(total_rows, end)
            for i in range(start0, end0 + 1):
                indices.add(i - 1)
        else:
            if token.isdigit():
                i1 = int(token)
                if 1 <= i1 <= total_rows:
                    indices.add(i1 - 1)
    return sorted(indices)

def estimate_processing_time(num_requests: int, rpm: int) -> str:
    """Estimate processing time based on rate limits"""
    minutes = num_requests / rpm
    if minutes < 1:
        return f"{int(minutes * 60)} seconds"
    elif minutes < 60:
        return f"{minutes:.1f} minutes"
    else:
        hours = minutes / 60
        return f"{hours:.1f} hours"

def main():
    st.set_page_config(
        page_title="OpenRouter Batch Processor",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header
    st.title("🚀 OpenRouter Batch Processor")
    st.markdown("**Professional batch processing for legal documents and text analysis**")
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Section
        with st.expander("🔑 API Authentication", expanded=True):
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Get your key from https://openrouter.ai/keys",
                key="api_key_input"
            )
            
            if api_key and not st.session_state.api_validated:
                if st.button("Validate API Key", key="validate_btn"):
                    with st.spinner("Validating..."):
                        temp_client = OpenRouterAPI(api_key)
                        valid, message = asyncio.run(temp_client.validate_api_key())
                        if valid:
                            st.session_state.api_validated = True
                            st.success(message)
                        else:
                            st.error(message)
            elif st.session_state.api_validated:
                st.success("✅ API key validated")
        
        # Rate Limiting - Updated with correct defaults
        with st.expander("⏱️ Rate Limiting", expanded=False):
            rpm_limit = st.slider(
                "Requests per Minute",
                min_value=1,
                max_value=30,
                value=DEFAULT_RPM_LIMIT,  # Now 20 RPM
                help="Free models: 20 RPM limit"
            )
            
            daily_limit = st.number_input(
                "Daily Request Limit",
                min_value=1,
                max_value=5000,
                value=DEFAULT_DAILY_LIMIT,
                help="Free models: 50/day with <$10 credits, 1000/day with $10+ credits"
            )

            request_timeout = st.number_input(
                "Request timeout (seconds)",
                min_value=30,
                max_value=600,
                value=int(st.session_state.request_timeout),
                step=10,
                help="Maximum time to wait for each OpenRouter response"
            )
            st.session_state.request_timeout = float(request_timeout)
        
        # Preset Configuration
        st.subheader("🎯 Preset Configuration")

        preset_name = st.text_input(
            "Preset Slug",
            placeholder="my-legal-extraction-preset",
            help="Enter the preset slug shown in your OpenRouter dashboard"
        )

        preset_method = st.selectbox(
            "Preset Reference",
            options=["field", "direct_reference"],
            format_func=lambda x: {
                "field": "Preset Field (recommended)",
                "direct_reference": "Direct Model Reference (@preset/name)"
            }[x],
            help="Choose how to reference your preset in the API call"
        )

        st.caption("Presets merge these overrides with the configuration stored on OpenRouter.")

        apply_overrides = False
        temperature: Optional[float] = None
        max_tokens: Optional[int] = None
        top_p: Optional[float] = None

        model = None  # Direct model selection disabled; presets are required

        with st.expander("🎛️ Preset Overrides (optional)", expanded=False):
            apply_overrides = st.checkbox(
                "Send manual parameter overrides",
                value=st.session_state.get('apply_overrides', False),
                help="Enable to override temperature, max tokens, or top_p in addition to preset defaults",
                key="apply_overrides"
            )

            if apply_overrides:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.05,
                    help="Lower values keep responses closer to the preset"
                )

                max_tokens = st.number_input(
                    "Max Output Tokens",
                    min_value=100,
                    max_value=28000,
                    value=24000,
                    step=100,
                    help="Upper bound for response length"
                )

                top_p = st.slider(
                    "Top P",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    help="Nucleus sampling parameter"
                )

        # Optional provider routing controls
        with st.expander("🧭 Provider Routing (optional)", expanded=False):
            st.caption("If left blank, the preset or default routing will be used.")
            provider_allow_str = st.text_input(
                "Allow providers (comma-separated)",
                placeholder="e.g., openai, anthropic, deepseek"
            )
            provider_deny_str = st.text_input(
                "Deny providers (comma-separated)",
                placeholder="e.g., openai-compatible"
            )
            provider_order_str = st.text_input(
                "Preferred order (comma-separated)",
                placeholder="e.g., deepseek, openai, anthropic"
            )
        
        # System Prompt
        with st.expander("📝 System Prompt (Optional)", expanded=False):
            system_prompt = st.text_area(
                "System Prompt",
                placeholder="You are a legal document analyzer. Extract key facts...",
                help="Instructions for the model (may be merged with preset system prompt)",
                height=100
            )
    
    # Main interface
    if not api_key:
        st.warning("⚠️ Please enter your OpenRouter API key in the sidebar to begin.")
        st.info("Get your free API key at: https://openrouter.ai/keys")
        return
    
    if not st.session_state.api_validated:
        st.info("🔍 Please validate your API key before proceeding.")
        return
    
    # File upload section
    st.header("📁 Data Input")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV with ID and content columns"
        )
    
    with col2:
        if st.button("📋 Use Sample Data", help="Load sample data for testing"):
            sample_data = pd.DataFrame({
                'id': ['doc_001', 'doc_002', 'doc_003'],
                'content': [
                    "This is a sample legal document clause about liability limitations...",
                    "Contract terms regarding payment schedules and penalties...",
                    "Jurisdiction and governing law provisions for international agreements..."
                ]
            })
            st.session_state.sample_data = sample_data
    
    # Handle data source
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            valid, message = validate_csv_data(df)
            if valid:
                st.success(f"✅ Loaded {len(df):,} rows from CSV")
            else:
                st.error(f"❌ {message}")
                return
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
            return
    elif 'sample_data' in st.session_state:
        df = st.session_state.sample_data
        st.info("📋 Using sample data")
    
    if df is not None:
        # Column selection
        st.subheader("🎯 Column Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            id_column = st.selectbox(
                "ID Column",
                options=df.columns.tolist(),
                index=0 if 'id' in df.columns else 0,
                help="Column with unique identifiers"
            )
        
        with col2:
            content_column = st.selectbox(
                "Content Column",
                options=df.columns.tolist(),
                index=1 if len(df.columns) > 1 else 0,
                help="Column with text to process"
            )
        
        # Data preview
        st.subheader("👀 Data Preview")
        
        # Show data stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            avg_length = df[content_column].astype(str).str.len().mean()
            st.metric("Avg Content Length", f"{avg_length:.0f} chars")
        with col3:
            st.metric("Columns", len(df.columns))
        with col4:
            duplicates = df[id_column].duplicated().sum()
            st.metric("Duplicate IDs", duplicates, delta=None if duplicates == 0 else "⚠️")
        
        # Show preview with expandable content
        preview_df = df[[id_column, content_column]].head(10).copy()
        preview_df[content_column] = preview_df[content_column].astype(str).str[:100] + "..."
        st.dataframe(preview_df, use_container_width=True)
        
        # Processing configuration
        st.header("🔧 Processing Configuration")

        # Row selection input (1-based)
        default_selection = f"1-{min(len(df), 50)}"
        row_selection_str = st.text_input(
            "Row selection (1-based)",
            value=st.session_state.row_selection or default_selection,
            help="Examples: 2-999, 5, 10-20. Ranges are inclusive and can be combined with commas.",
        )
        st.session_state.row_selection = row_selection_str
        selected_indices = parse_row_selection(row_selection_str, len(df))
        st.session_state.selected_row_indices = selected_indices

        if st.session_state.last_row_selection != row_selection_str:
            st.session_state.pending_indices = []
            st.session_state.selection_snapshot = selected_indices.copy()
            st.session_state.total_selected = len(selected_indices)
            st.session_state.processing_cursor = 0
            st.session_state.pause_requested = False
            st.session_state.processing_active = False
            st.session_state.last_row_selection = row_selection_str

        if not st.session_state.selection_snapshot and selected_indices:
            st.session_state.selection_snapshot = selected_indices.copy()

        st.caption("Separate multiple ranges with commas, e.g. 5-12, 18, 22-30.")

        processed_rows = set()
        for row in st.session_state.get('processing_results', []):
            row_no = row.get('row_number')
            if isinstance(row_no, (int, float)):
                processed_rows.add(int(row_no) - 1)

        selection_base = st.session_state.selection_snapshot or selected_indices
        processed_count = sum(1 for idx in selection_base if idx in processed_rows)
        remaining_count = max(0, len(selection_base) - processed_count)
        st.session_state.total_selected = len(selection_base)
        st.session_state.processing_cursor = processed_count

        selected_count = len(selection_base)

        pending_for_calc = len(st.session_state.pending_indices)
        if pending_for_calc > 0:
            remaining_for_batch = pending_for_calc
        elif remaining_count > 0:
            remaining_for_batch = remaining_count
        else:
            remaining_for_batch = selected_count or len(df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Selected", selected_count)
        with col2:
            st.metric("Processed", processed_count)
        with col3:
            st.metric("Remaining", remaining_count)
        with col4:
            # Chunk size for next run
            max_chunk = max(1, min(remaining_for_batch, 200))
            default_chunk = min(st.session_state.chunk_size, max_chunk) if st.session_state.chunk_size else min(10, max_chunk)
            chunk_size = st.number_input(
                "Batch size (rows per run)",
                min_value=1,
                max_value=max_chunk,
                value=default_chunk,
                help="Number of rows to submit in the next batch"
            )
            st.session_state.chunk_size = int(chunk_size)

        # Estimated time for next batch
        next_batch = min(st.session_state.chunk_size, max(1, remaining_for_batch))
        estimated_time = estimate_processing_time(max(1, next_batch), rpm_limit)
        st.caption(f"⏱️ Estimated time for next batch: {estimated_time}")

        if selected_count > 0:
            overall_ratio = processed_count / selected_count if selected_count else 0
            overall_progress = st.progress(overall_ratio)
            st.caption(f"Overall progress: {processed_count}/{selected_count} rows completed")
        
        # Processing controls
        st.subheader("🚀 Processing Controls")

        start_col, pause_col, reset_col, auto_col = st.columns([1, 1, 1, 1.2])

        start_disabled = selected_count == 0 or not preset_name
        with start_col:
            if st.button("▶️ Start / Continue", type="primary", disabled=start_disabled):
                if start_disabled:
                    if selected_count == 0:
                        st.warning("Select at least one row before starting.")
                    else:
                        st.error("Enter a preset slug to proceed.")
                else:
                    queue = [idx for idx in selection_base if idx not in processed_rows]
                    if not queue:
                        st.info("All selected rows are already processed. Reset the queue to run again.")
                        append_debug_log(
                            f"[{datetime.now().strftime('%H:%M:%S')}] START requested but queue already empty"
                        )
                    else:
                        st.session_state.pending_indices = queue
                        st.session_state.selection_snapshot = selection_base.copy()
                        st.session_state.total_selected = len(selection_base)
                        st.session_state.processing_cursor = len(selection_base) - len(queue)
                        st.session_state.processing_active = True
                        st.session_state.pause_requested = False
                        append_debug_log(
                            f"[{datetime.now().strftime('%H:%M:%S')}] START processing {len(queue)} queued rows (chunk size={st.session_state.chunk_size})"
                        )
                        st.rerun()

        with pause_col:
            if st.button("⏸ Pause", disabled=not st.session_state.processing_active):
                st.session_state.processing_active = False
                st.session_state.pause_requested = True
                append_debug_log(f"[{datetime.now().strftime('%H:%M:%S')}] PAUSED queue")
                st.rerun()

        with reset_col:
            if st.button("♻️ Reset Queue"):
                st.session_state.processing_results = []
                st.session_state.pending_indices = []
                st.session_state.processing_active = False
                st.session_state.processing_cursor = 0
                st.session_state.pause_requested = False
                st.session_state.selection_snapshot = selected_indices.copy()
                st.session_state.total_selected = len(selected_indices)
                st.session_state.debug_log = []
                st.rerun()

        with auto_col:
            auto_continue = st.checkbox(
                "Auto-continue batches",
                value=st.session_state.auto_resume,
                help="Run batches back-to-back until the queue is empty."
            )
            st.session_state.auto_resume = auto_continue

        queue_preview = st.session_state.pending_indices or [idx for idx in selection_base if idx not in processed_rows]
        if queue_preview:
            preview_ids = df.iloc[queue_preview[:min(5, len(queue_preview))]][id_column].astype(str).tolist()
            st.caption(f"Next rows in queue: {', '.join(preview_ids)}")

        with st.expander("📟 Debug Console", expanded=False):
            debug_output = "\n".join(st.session_state.get('debug_log', [])) or "No log entries yet."
            st.code(debug_output, language="text")
            if st.button("🧹 Clear Debug Log", key="clear_debug_log"):
                st.session_state.debug_log = []
                st.rerun()
        
        # Processing execution
        if st.session_state.processing_active:
            st.subheader("⚡ Processing in Progress")
            if not preset_name:
                st.error("Preset slug is required to continue processing.")
                st.session_state.processing_active = False
                pending_indices = []
            else:
                pending_indices = st.session_state.pending_indices

            if not pending_indices:
                st.info("Queue is empty—nothing to process.")
                st.session_state.processing_active = False
                append_debug_log(f"[{datetime.now().strftime('%H:%M:%S')}] Queue empty; nothing to process")
            else:
                api_client = OpenRouterAPI(api_key, rpm_limit, daily_limit)
                chunk_indices = list(pending_indices[:st.session_state.chunk_size])

                if not chunk_indices:
                    st.info("No rows remaining to process.")
                    st.session_state.processing_active = False
                    append_debug_log(f"[{datetime.now().strftime('%H:%M:%S')}] No rows remaining for this batch")
                else:
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(current, total, status_msg):
                            progress = current / total if total > 0 else 0
                            progress_bar.progress(progress)
                            status_text.text(f"{status_msg} ({current}/{total})")

                        batch_params: Dict[str, Any] = {}
                        if apply_overrides:
                            batch_params = {
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "top_p": top_p
                            }

                        providers_obj: Dict[str, Any] = {}
                        allow_list = [p.strip() for p in provider_allow_str.split(",") if p.strip()] if 'provider_allow_str' in locals() else []
                        deny_list = [p.strip() for p in provider_deny_str.split(",") if p.strip()] if 'provider_deny_str' in locals() else []
                        order_list = [p.strip() for p in provider_order_str.split(",") if p.strip()] if 'provider_order_str' in locals() else []

                        if allow_list:
                            providers_obj["allow"] = allow_list
                        if deny_list:
                            providers_obj["deny"] = deny_list
                        if order_list:
                            providers_obj["order"] = order_list

                        if providers_obj:
                            batch_params["providers"] = providers_obj

                        try:
                            effective_system_prompt = system_prompt
                            asyncio.run(
                                process_batch(
                                    api_client=api_client,
                                    df=df,
                                    id_column=id_column,
                                    content_column=content_column,
                                    preset=preset_name,
                                    model=model,
                                    system_prompt=effective_system_prompt,
                                    row_indices=chunk_indices,
                                    batch_params=batch_params,
                                    preset_method=preset_method,
                                    rpm_limit=rpm_limit,
                                    max_concurrency=min(8, rpm_limit),
                                    request_timeout=st.session_state.request_timeout,
                                    progress_callback=update_progress
                                )
                            )

                            progress_bar.progress(1.0)
                            status_text.text("✅ Batch completed!")

                            del st.session_state.pending_indices[:len(chunk_indices)]
                            remaining_after = len(st.session_state.pending_indices)
                            st.session_state.processing_cursor = st.session_state.total_selected - remaining_after

                            if st.session_state.auto_resume and st.session_state.pending_indices:
                                st.session_state.processing_active = True
                                st.rerun()
                            else:
                                st.session_state.processing_active = False
                                append_debug_log(
                                    f"[{datetime.now().strftime('%H:%M:%S')}] Batch finished. Remaining queue: {remaining_after}"
                                )

                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            st.session_state.processing_active = False
                            append_debug_log(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during batch: {str(e)}")
        
        # Results display
        if st.session_state.processing_results:
            st.header("📊 Results")
            
            results_df = pd.DataFrame(st.session_state.processing_results)

            if 'row_number' in results_df.columns:
                results_df = results_df.sort_values('row_number')
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            successful = results_df['success'].sum()
            failed = len(results_df) - successful
            
            with col1:
                st.metric("Processed (selection)", processed_count)
            with col2:
                st.metric("Successful", successful, delta=f"{successful/len(results_df)*100:.1f}%")
            with col3:
                st.metric("Failed", failed, delta=f"{failed/len(results_df)*100:.1f}%" if failed > 0 else None)
            with col4:
                if 'total_tokens' in results_df.columns:
                    total_tokens = results_df['total_tokens'].sum()
                    st.metric("Total Tokens", f"{total_tokens:,}")
            
            # Results table with better formatting
            display_columns = []
            if 'row_number' in results_df.columns:
                display_columns.append('row_number')
            display_columns.extend([id_column, 'success', 'model_used', 'markdown_file'])
            if 'total_tokens' in results_df.columns:
                display_columns.append('total_tokens')
            if failed > 0:
                display_columns.append('error')
            
            st.dataframe(
                results_df[display_columns],
                use_container_width=True,
                column_config={
                    'success': st.column_config.CheckboxColumn('Success'),
                    'total_tokens': st.column_config.NumberColumn('Tokens', format='%d')
                }
            )
            
            # Download section
            st.subheader("💾 Download Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # CSV download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📊 Download CSV",
                    csv,
                    f"results_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON download
                json_data = results_df.to_json(orient='records', indent=2)
                st.download_button(
                    "📄 Download JSON",
                    json_data,
                    f"results_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    "application/json",
                    use_container_width=True
                )
            
            with col3:
                # ZIP download
                output_dir = os.path.join(OUTPUT_DIR_BASE, f"session_{st.session_state.session_id}")
                if os.path.exists(output_dir):
                    zip_data = create_zip_download(output_dir)
                    st.download_button(
                        "📁 Download ZIP",
                        zip_data,
                        f"markdown_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                        "application/zip",
                        use_container_width=True
                    )

            with col4:
                # Plain-text responses only
                responses_txt = build_responses_text(results_df, id_column)
                st.download_button(
                    "📝 Download TXT",
                    responses_txt,
                    f"responses_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain",
                    use_container_width=True,
                    disabled=responses_txt.startswith("No responses available")
                )
            
            # Error analysis (if any)
            if failed > 0:
                with st.expander(f"❌ Error Analysis ({failed} errors)", expanded=False):
                    error_df = results_df[~results_df['success']][[id_column, 'error']]
                    st.dataframe(error_df, use_container_width=True)
    
    # Footer with tips - Updated with current information
    with st.expander("💡 Usage Tips & Best Practices"):
        st.markdown("""
        ### For Optimal Results:
        
        **Preset vs Direct Model:**
        - ✅ **Use Presets** for production workflows - configure temperature, model, and system prompts in OpenRouter
        - ✅ **Direct Model** for testing and experimentation
        - ✅ **Parameter Merging**: When using presets, your request parameters will be shallow-merged with preset configuration
        
        **Enhanced Preset Support:**
        - 🎯 **Preset Field Method** (recommended): Uses `{"preset": "preset-name"}` in API request
        - 🎯 **Direct Reference Method**: Uses `"@preset/preset-name"` as the model parameter
        
        **Legal Document Processing:**
        - 📋 Use temperature 0.1-0.2 for factual extraction
        - 📋 DeepSeek R1 Distill 70B is excellent for reasoning and legal analysis
        - 📋 Qwen 2.5 Coder 32B for structured extraction
        - 📋 Grok 4 Fast for large documents (2M context)
        - 📋 Include clear instructions in system prompt or preset
        
        **Updated Rate Limiting:**
        - ⏱️ Free models: **20 RPM** (updated from 5 RPM)
        - ⏱️ Daily limits: **50 requests/day** with <$10 credits, **1000/day** with $10+ credits
        - ⏱️ Process during off-peak hours for better performance
        
        **File Management:**
        - 💾 All outputs are saved automatically
        - 💾 Session-based organization prevents conflicts
        - 💾 Download results in multiple formats
        """)

if __name__ == "__main__":
    main()
