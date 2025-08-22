import types
import sys
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

# We import the module under test. If its filename is not "test_main.py" but "main.py",
# adjust the import below to match your actual module name. The content provided in the PR
# shows functions `main()` and `generate_content()` along with `if __name__ == "__main__": main()`,
# which implies a top-level module. We'll import it dynamically by filename if needed.
#
# Default attempt: try "main" then fallback to "test_main" (as provided tag name).
try:
    import main as mod
except Exception:
    import importlib.util, pathlib
    mname_candidates = ["test_main", "app.main", "src.main"]
    _imported = False
    for _mn in mname_candidates:
        try:
            mod = __import__(_mn, fromlist=["*"])
            _imported = True
            break
        except Exception:
            continue
    if not _imported:
        # Best-effort: try to load a file literally named "test_main.py" as module "mod"
        if pathlib.Path("test_main.py").exists():
            spec = importlib.util.spec_from_file_location("mod", "test_main.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
        else:
            raise

# Test framework note:
# Using pytest for concise, readable tests with fixtures and monkeypatching.

@contextmanager
def patched_argv(argv):
    """
    Context manager that temporarily replaces sys.argv with a provided list.
    
    Parameters:
        argv (list[str]): Sequence to use as sys.argv within the context.
    
    Usage:
        Use in a `with` statement to run code with a mocked command-line argument list; the original sys.argv is restored on exit, even if an exception is raised.
    """
    old = sys.argv[:]
    sys.argv = argv[:]
    try:
        yield
    finally:
        sys.argv = old

class DummyUsageMetadata:
    def __init__(self, prompt=0, resp=0):
        """
        Initialize usage metadata.
        
        Parameters:
            prompt (int): Number of tokens consumed by the prompt.
            resp (int): Number of tokens consumed by generated candidates.
        """
        self.prompt_token_count = prompt
        self.candidates_token_count = resp

class DummyPart:
    def __init__(self, text=None, function_response=None):
        # Mimic google.genai.types.Part attributes used by code
        """
        Initialize a dummy Part used to emulate google.genai.types.Part in tests.
        
        Parameters:
            text (str | None): The textual content of the part.
            function_response (Optional[DummyFunctionResponse]): If present, represents a function call result attached to this part (wraps the underlying value); otherwise None.
        """
        self.text = text
        self.function_response = function_response

class DummyFunctionResponse:
    def __init__(self, response):
        """
        Initialize the object with a preconfigured response.
        
        Parameters:
            response: Dummy response object that will be returned by the instance (stored on self.response).
        """
        self.response = response

class DummyContent:
    def __init__(self, role="user", parts=None):
        """
        Initialize a content block with a role and an ordered list of parts.
        
        Parameters:
            role (str): Role of the content (e.g., "user", "assistant", "tool"). Defaults to "user".
            parts (Sequence[Part] | None): Ordered list of parts making up the content; if None, an empty list is used.
        """
        self.role = role
        self.parts = parts or []

class DummyCandidate:
    def __init__(self, content):
        """
        Initialize the instance with a content block.
        
        Parameters:
            content (google.genai.types.Content | DummyContent): Content object representing a candidate response or message.
        """
        self.content = content

class DummyResponse:
    def __init__(self, *, text=None, candidates=None, function_calls=None, usage_prompt=0, usage_resp=0):
        """
        Initialize a dummy response object representing a model's output.
        
        Parameters:
            text (str | None): Final textual reply from the model, if any.
            candidates (list | None): Sequence of candidate objects (e.g., content candidates); defaults to an empty list.
            function_calls (list | None): Sequence of function-call descriptors returned by the model; defaults to an empty list.
            usage_prompt (int): Token count attributed to the prompt portion of the request (stored in usage_metadata).
            usage_resp (int): Token count attributed to the model's response (stored in usage_metadata).
        
        This constructor sets the instance attributes `text`, `candidates`, `function_calls`, and creates a `usage_metadata`
        object (DummyUsageMetadata) from the provided token counts.
        """
        self.text = text
        self.candidates = candidates or []
        self.function_calls = function_calls or []
        self.usage_metadata = DummyUsageMetadata(usage_prompt, usage_resp)

class DummyModels:
    def __init__(self, response):
        """
        Initialize the models holder with a prebuilt response.
        
        Parameters:
            response: A response object (e.g., DummyResponse) that will be returned by generate_content().
        """
        self._response = response
    def generate_content(self, **kwargs):
        """
        Return the stored dummy response for testing.
        
        This method simulates a content-generation API call by returning the preconfigured response object.
        Any keyword arguments passed are ignored.
        
        Returns:
            The dummy response object previously assigned to the instance (typically a DummyResponse).
        """
        return self._response

class DummyClient:
    def __init__(self, response):
        """
        Initialize the dummy client with a preset response.
        
        Parameters:
            response: The DummyResponse (or compatible object) that will be returned by this client's models.generate_content calls.
        """
        self.models = DummyModels(response)

def make_types_namespace():
    # Build a lightweight stand-in for google.genai.types with Content and Part and GenerateContentConfig
    """
    Create a lightweight stand-in for the `google.genai.types` namespace used in tests.
    
    Returns a SimpleNamespace with:
    - Part: class used to represent a content part (DummyPart).
    - Content: class used to represent content blocks (DummyContent).
    - GenerateContentConfig: small config class with signature __init__(self, tools=None, system_instruction=None).
    
    This namespace is intended for use in tests to mock the real `google.genai.types` module.
    """
    ns = types.SimpleNamespace()
    ns.Part = DummyPart
    ns.Content = DummyContent
    class GenerateContentConfig:
        def __init__(self, tools=None, system_instruction=None):
            """
            Initialize a GenerateContentConfig.
            
            Parameters:
                tools (Optional[list]): Optional sequence of tool definitions (e.g., callables or descriptors) available to the content generator. If None, no tools are provided.
                system_instruction (Optional[str]): Optional system-level instruction or prompt that guides the generator's behavior.
            """
            self.tools = tools
            self.system_instruction = system_instruction
    ns.GenerateContentConfig = GenerateContentConfig
    return ns

@pytest.fixture(autouse=True)
def patch_types(monkeypatch):
    # Replace imported google.genai.types in the module with our dummy types
    """
    Autouse pytest fixture that replaces the module's google.genai.types with a lightweight dummy types namespace used by the tests.
    
    This ensures code under test imports predictable stand-ins (Part, Content, GenerateContentConfig, etc.) so tests can run without the real google.genai.types package or networked GenAI services.
    """
    monkeypatch.setattr(mod, "types", make_types_namespace())

@pytest.fixture
def capprinted(capsys):
    # helper to retrieve printed output conveniently
    """
    Return a callable that reads and returns captured stdout and stderr.
    
    Given pytest's `capsys` fixture, returns a zero-argument function which, when called,
    invokes `capsys.readouterr()` and returns a tuple (out, err) of captured stdout and stderr.
    """
    def _read():
        out, err = capsys.readouterr()
        return out, err
    return _read

def test_generate_content_returns_text_when_no_function_calls(monkeypatch, capprinted):
    # Arrange: response with text and candidates and no function_calls
    candidate_content = DummyContent(role="model", parts=[DummyPart(text="candidate")])
    response = DummyResponse(
        text="final answer",
        candidates=[DummyCandidate(candidate_content)],
        function_calls=[],
        usage_prompt=12,
        usage_resp=34,
    )
    client = DummyClient(response)
    messages = [DummyContent(role="user", parts=[DummyPart(text="hello")])]

    # Verbose True to exercise token printing and candidate append
    result = mod.generate_content(client, messages, True)

    # Assert
    assert result == "final answer"
    # Candidate content appended to messages
    assert any(m is candidate_content for m in messages), "candidate content should be appended to messages"
    out, _ = capprinted()
    assert "Prompt tokens: 12" in out
    assert "Response tokens: 34" in out

def test_generate_content_calls_functions_and_appends_tool_message(monkeypatch, capprinted):
    # Arrange: response with function_calls triggers call_function
    fc_part = SimpleNamespace(name="do_something", args={"x": 1})
    response = DummyResponse(
        text=None,
        candidates=[],
        function_calls=[fc_part],
        usage_prompt=1,
        usage_resp=2,
    )
    client = DummyClient(response)
    messages = [DummyContent(role="user", parts=[DummyPart(text="go")])]

    # Mock call_function to return an object with parts[0].function_response.response present
    class FuncResult:
        def __init__(self):
            """
            Initialize the instance with a single part containing a function response.
            
            The constructor creates self.parts as a list with one DummyPart whose function_response
            is a DummyFunctionResponse wrapping the string "OK". This provides a ready-made
            non-empty function call result for use in tests.
            """
            self.parts = [DummyPart(function_response=DummyFunctionResponse(response="OK"))]
    monkeypatch.setattr(mod, "call_function", lambda part, verbose: FuncResult())

    # Act
    ret = mod.generate_content(client, messages, True)

    # Assert: function path returns None but appends a tool message with the function response part
    assert ret is None
    assert any(msg for msg in messages if msg.role == "tool" and msg.parts and isinstance(msg.parts[0], DummyPart))
    out, _ = capprinted()
    assert "-> OK" in out

def test_generate_content_raises_on_empty_function_call_result(monkeypatch):
    # Arrange: function call but call_function returns empty parts
    fc_part = SimpleNamespace(name="do_something", args={})
    response = DummyResponse(function_calls=[fc_part])
    client = DummyClient(response)
    messages = []

    class BadFuncResult:
        def __init__(self):
            """
            Initialize a function-response holder with an empty parts list.
            
            The empty `parts` list indicates no content has been added; callers should treat an instance with no parts as an invalid/empty function call result.
            """
            self.parts = []  # empty -> should raise

    monkeypatch.setattr(mod, "call_function", lambda part, verbose: BadFuncResult())

    with pytest.raises(Exception, match="empty function call result"):
        mod.generate_content(client, messages, False)

def test_generate_content_raises_when_no_function_responses(monkeypatch):
    # Arrange: function call returns non-empty parts but without function_response
    fc_part = SimpleNamespace(name="f", args={})
    response = DummyResponse(function_calls=[fc_part])
    client = DummyClient(response)
    messages = []

    class NoFunctionResponse:
        def __init__(self):
            """
            Initialize a response whose single part contains no function_response.
            
            This constructs an instance with .parts set to a one-element list containing a DummyPart with function_response=None. Used in tests to simulate an empty/invalid function call result that should cause the caller to raise an error.
            """
            self.parts = [DummyPart(function_response=None)]  # no function_response object -> triggers raise

    monkeypatch.setattr(mod, "call_function", lambda part, verbose: NoFunctionResponse())

    with pytest.raises(Exception, match="empty function call result"):
        mod.generate_content(client, messages, False)

def test_main_exits_with_usage_when_no_args(monkeypatch, capprinted):
    # Prevent actual dotenv and client creation; we won't reach those lines due to early exit.
    monkeypatch.setattr(mod, "load_dotenv", lambda: None)
    with patched_argv(["main.py"]):
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert e.value.code == 1
    out, _ = capprinted()
    assert "Usage: python main.py" in out

def test_main_respects_max_iters_and_exits(monkeypatch, capprinted):
    # Arrange: set MAX_ITERS small and have generate_content return falsy to keep looping until limit
    monkeypatch.setattr(mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    # Dummy client returned by genai.Client
    class DummyGenAIClient:
        pass
    monkeypatch.setattr(mod.genai, "Client", lambda api_key=None: DummyGenAIClient())
    # Force MAX_ITERS = 3
    monkeypatch.setattr(mod, "MAX_ITERS", 3)
    # Make generate_content return None each time to never produce a final_response
    call_counter = {"n": 0}
    def fake_generate_content(client, messages, verbose):
        """
        Test helper that mimics `generate_content` by incrementing an external call counter and returning None.
        
        This function is a stand-in used in tests: it does not use its arguments but increments call_counter["n"] as a side effect to record invocations.
        
        Parameters:
            client, messages, verbose: Ignored; present only to match the real `generate_content` signature.
        
        Returns:
            None
        """
        call_counter["n"] += 1
        return None
    monkeypatch.setattr(mod, "generate_content", fake_generate_content)
    with patched_argv(["main.py", "Do work"]):
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert e.value.code == 1
    out, _ = capprinted()
    assert "Maximum iterations (3) reached." in out
    # generate_content should have been called 3 times inside the loop and NOT the extra call after loop (since exit happened)
    assert call_counter["n"] == 3

def test_main_breaks_on_final_response_and_calls_generate_content_once_more(monkeypatch, capprinted):
    # Arrange: make first loop iteration return a truthy final_response, then verify extra call after loop
    monkeypatch.setattr(mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    class DummyGenAIClient:
        pass
    monkeypatch.setattr(mod.genai, "Client", lambda api_key=None: DummyGenAIClient())
    monkeypatch.setattr(mod, "MAX_ITERS", 5)
    calls = {"n": 0}
    def fake_generate_content(client, messages, verbose):
        """
        Test helper that simulates generate_content behavior for unit tests.
        
        Increments the shared `calls["n"]` counter as a side effect. Returns the string "ok" on the first invocation (when the counter becomes 1) and "ignored" on subsequent invocations. Intended for use in tests that assert loop and retry behavior.
        
        Returns:
            str: "ok" for the first call, "ignored" thereafter.
        """
        calls["n"] += 1
        # First call returns a final response string; subsequent call can return another string or None
        return "ok" if calls["n"] == 1 else "ignored"
    monkeypatch.setattr(mod, "generate_content", fake_generate_content)
    with patched_argv(["main.py", "Do work", "--verbose"]):
        mod.main()
    out, _ = capprinted()
    # Should print 'Final response:' and the content, then break; then call generate_content once more after loop
    assert "Final response:" in out
    assert "ok" in out
    assert calls["n"] == 2, "generate_content should be called once to get final response, then once more after loop"

def test_main_prints_user_prompt_when_verbose(monkeypatch, capprinted):
    monkeypatch.setattr(mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    class DummyGenAIClient:
        pass
    monkeypatch.setattr(mod.genai, "Client", lambda api_key=None: DummyGenAIClient())
    # Make generate_content return a response immediately to exit loop quickly
    monkeypatch.setattr(mod, "generate_content", lambda c, m, v: "done")
    with patched_argv(["main.py", "Quick", "test", "--verbose"]):
        mod.main()
    out, _ = capprinted()
    assert "User prompt: Quick test" in out

def test_main_handles_exceptions_and_continues(monkeypatch, capprinted):
    # Verify that exceptions in generate_content are caught and printed, then next iteration proceeds.
    monkeypatch.setattr(mod, "load_dotenv", lambda: None)
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    class DummyGenAIClient:
        pass
    monkeypatch.setattr(mod.genai, "Client", lambda api_key=None: DummyGenAIClient())
    monkeypatch.setattr(mod, "MAX_ITERS", 3)
    calls = {"n": 0}
    def flaky(client, messages, verbose):
        """
        Test helper that simulates a flaky operation for use in tests.
        
        Increments the global `calls["n"]` counter each call. Behavior by invocation count:
        - 1st call: raises RuntimeError("boom").
        - 2nd call: returns None.
        - 3rd and subsequent calls: returns the string "ok".
        
        Parameters:
            client: Ignored. Present to match the real function signature.
            messages: Ignored. Present to match the real function signature.
            verbose: Ignored. Present to match the real function signature.
        
        Returns:
            str | None: "ok" on the third and later calls, None on the second call.
        
        Raises:
            RuntimeError: Always raised on the first invocation.
        """
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        elif calls["n"] == 2:
            return None
        else:
            return "ok"
    monkeypatch.setattr(mod, "generate_content", flaky)
    with patched_argv(["main.py", "Do work"]):
        mod.main()
    out, _ = capprinted()
    assert "Error in generate_content: boom" in out
    # Should finish with a final response then exit loop, then call extra once more
    assert calls["n"] == 4, "1st raises, 2nd None, 3rd ok (break), 4th extra call after loop"