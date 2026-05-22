var s=null;async function p(){s||(self.postMessage({id:"init",type:"progress",payload:{progress:10,message:"Loading Pyodide JS..."}}),importScripts("https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"),self.postMessage({id:"init",type:"progress",payload:{progress:50,message:"Initializing Pyodide Runtime..."}}),s=await self.loadPyodide({indexURL:"https://cdn.jsdelivr.net/pyodide/v0.25.0/full/"}),self.postMessage({id:"init",type:"progress",payload:{progress:100,message:"Pyodide Ready"}}))}self.onmessage=async i=>{let{id:r,type:t,payload:a}=i.data;try{if(t==="INIT"){await p(),self.postMessage({id:r,type:"success",payload:!0});return}if(!s)throw new Error("Pyodide is not initialized");if(t==="PARSE_ONNXSCRIPT"){let n=`
import sys
import traceback

def execute_user_script():
    try:
        # Dummy stub for onnxscript parsing since actual onnxscript package might be large.
        # This mocks extracting an ONNX protobuf string.
        user_script = """${a.replace(/"/g,'\\"')}"""
        if "SyntaxError" in user_script:
            raise SyntaxError("Simulated syntax error in line 2")
        return b"MOCK_ONNX_PROTOBUF_BYTES".hex()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        return {
            "error": True,
            "message": str(e),
            "traceback": traceback.format_exc()
        }

execute_user_script()
`,e=await s.runPythonAsync(n);if(typeof e=="object"&&e!==null&&e.error)throw new Error(JSON.stringify(e));self.postMessage({id:r,type:"success",payload:e})}else throw new Error(`Unsupported pyodide task: ${t}`)}catch(o){self.postMessage({id:r,type:"error",error:o.message||String(o)})}};
