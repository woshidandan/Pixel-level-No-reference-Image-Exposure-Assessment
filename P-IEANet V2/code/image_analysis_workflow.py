import base64
import json
import sys
import os
import uuid
from typing import List, Dict, Any, Optional, Literal
import httpx
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from grounded_sam_tool import GroundedSamTool
from exposure_tool import ExposureTool
from dual_grounding import DualGroundingBuilder
from rule_rag import RuleRAGRetriever

# Try to import OpenAI (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not installed. Install with: pip install openai")

def encode_image_to_base64(file_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class ImageAnalysisWorkflow:
    """
    Improved image analysis workflow using multi-turn conversations.
    
    This workflow:
    1. First turn: Segment objects using Grounded-SAM
    2. Second turn: Evaluate exposure of segmented objects
    3. Third turn: Provide comprehensive analysis (supports both Llama Stack and OpenAI models)
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:8321",
                 openai_api_key: Optional[str] = None,
                 openai_model: str = "gpt-4o",
                 rules_path: Optional[str] = None):
        """
        Initialize the improved image analysis workflow.
        
        Args:
            base_url: Llama Stack server base URL
            openai_api_key: OpenAI API key (optional, can also be set via OPENAI_API_KEY env var)
            openai_model: OpenAI model to use for comprehensive analysis (default: gpt-4o)
            rules_path: Optional JSON rule library for Rule RAG
        """
        self.base_url = base_url
        self.timeout = httpx.Timeout(500.0)
        self.http_client = httpx.Client(timeout=self.timeout)
        
        # Initialize Llama Stack client
        self.client = LlamaStackClient(
            base_url=base_url,
            http_client=self.http_client
        )
        
        # Initialize OpenAI client (optional)
        self.openai_client = None
        self.openai_model = openai_model
        if OPENAI_AVAILABLE:
            # Try to initialize OpenAI client
            api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                # Get base_url from environment variable
                openai_base_url = os.environ.get("OPENAI_BASE_URL")
                
                # Create httpx client for OpenAI with longer timeout (5 minutes)
                openai_http_client = httpx.Client(timeout=httpx.Timeout(300.0))
                
                # Initialize with custom base_url if provided
                if openai_base_url:
                    self.openai_client = OpenAI(
                        api_key=api_key, 
                        base_url=openai_base_url,
                        http_client=openai_http_client
                    )
                    print(f"✅ OpenAI client initialized with custom base URL: {openai_base_url}")
                else:
                    self.openai_client = OpenAI(
                        api_key=api_key,
                        http_client=openai_http_client
                    )
                    print(f"✅ OpenAI client initialized with official API")
                print(f"   Model: {openai_model}")
                print(f"   Timeout: 300 seconds (5 minutes)")
            else:
                print("⚠️  OpenAI API key not provided. Set OPENAI_API_KEY env var or pass openai_api_key parameter.")
        
        # Initialize tools
        self.grounded_sam_tool = GroundedSamTool()
        self.exposure_tool = ExposureTool()
        self.rule_rag = RuleRAGRetriever(rules_path=rules_path)
        self.dual_grounding_builder = DualGroundingBuilder()
        
        # Create specialized agents for each step
        self.segmentation_agent = self._create_segmentation_agent()
        self.segmentation_executor_agent = self._create_segmentation_executor_agent()
        self.exposure_agent = self._create_exposure_agent()
        self.analysis_agent = self._create_analysis_agent()
    
    def _create_segmentation_agent(self) -> Agent:
        """Create agent specialized for object segmentation."""
        return Agent(
            self.client,
            model="llama4:scout",
            instructions="""
You are a specialized object segmentation assistant. Your job is to segment objects from images using the grounded_sam_segment tool.

CRITICAL RULES:
- Your ONLY job is to identify objects in the image and create text prompts for segmentation
- The text_prompt MUST be specific object names or descriptions, NOT questions or analysis
- Based on the user's request and what you see in the image, identify which objects need to be segmented
- Once you identify the objects, you MUST use the grounded_sam_segment tool to get segmentation results
- When calling the tool, use EXACTLY this format: [grounded_sam_segment(image_path="...", text_prompt="...")]
- IMPORTANT: Use the actual image path that was provided in the user's request, NOT a placeholder like "path_to_your_image.jpg"
- If you want to use a tool, output ONLY the tool call format, nothing else
- The text_prompt parameter supports multiple objects separated by commas (e.g., "bear, tree", "person, car, building")
- When analyzing complex scenes with multiple important objects, use comma-separated format to segment all relevant objects in one call

CORRECT TEXT PROMPT EXAMPLES:
✅ "person" - for a single person
✅ "man, woman" - for multiple people
✅ "person, car, building" - for multiple objects
✅ "the person, the car" - with articles
✅ "face, body" - for body parts
✅ "background, foreground" - for scene elements

INCORRECT TEXT PROMPT EXAMPLES:
❌ "Which person has better exposure?" - This is a question, not object names
❌ "Analyze the exposure level" - This is analysis, not object names
❌ "What objects are in this image?" - This is a question, not object names

Examples of correct output:
Single object: [grounded_sam_segment(image_path="/path/to/image.jpg", text_prompt="person")]
Multiple objects: [grounded_sam_segment(image_path="/path/to/image.jpg", text_prompt="man, woman")]
Multiple objects: [grounded_sam_segment(image_path="/path/to/image.jpg", text_prompt="person, car, building")]

""",
            tools=[self.grounded_sam_tool],
            enable_session_persistence=False,
        )
    
    def _create_exposure_agent(self) -> Agent:
        """Create agent specialized for exposure evaluation."""
        return Agent(
            self.client,
            model="llama4:scout",
            instructions="""
You are a specialized exposure evaluation assistant. Your job is to evaluate the exposure of segmented images using the evaluate_exposure tool.

CRITICAL RULES:
- You MUST use the evaluate_exposure tool to get exposure information.
- When calling the tool, use EXACTLY this format: [evaluate_exposure(image_path="...")]
- CRITICAL: Do NOT output ANY explanation, thinking, or other text before or after the tool call
- If you want to use a tool, output ONLY the tool call format, nothing else.
- Use the segmented image paths provided by the user, NOT the original image.
- This is the ONLY way to get accurate exposure analysis.
- IMPORTANT: Use the actual image path that was provided in the user's request, NOT a placeholder like "path_to_your_image.jpg"

Example of correct output:
[evaluate_exposure(image_path="/path/to/segmented_image.jpg")]

NOT like this:
I will evaluate the exposure of this image...
[evaluate_exposure(image_path="/path/to/segmented_image.jpg")]
The evaluation was successful...

After getting exposure results, briefly summarize the exposure scores and what they mean.
""",
            tools=[self.exposure_tool],
            enable_session_persistence=False,
        )
    
    def _create_segmentation_executor_agent(self) -> Agent:
        """Create agent specialized for executing segmentation tool calls."""
        return Agent(
            self.client,
            model="llama4:scout",
            instructions="""
You are a segmentation tool executor. Your job is to execute the grounded_sam_segment tool based on the tool call provided in the user's message.

CRITICAL RULES:
- You will receive a tool call format from the previous analysis step
- Extract the image_path and text_prompt from the tool call
- Execute the grounded_sam_segment tool with those exact parameters
- Execute the tool EXACTLY ONCE - no more, no less
- Do NOT add any explanation, thinking, or other text
- Output ONLY the tool call in the exact format provided
- Use the grounded_sam_segment tool to perform the segmentation
- After the tool executes successfully, STOP - do not call it again
- DO NOT repeat the tool call
- DO NOT call the tool multiple times
- The tool supports multiple objects in text_prompt (e.g., "person, car, building")

Example input: [grounded_sam_segment(image_path='/path/to/image.jpg', text_prompt='person, car, building')]
Example output: [grounded_sam_segment(image_path='/path/to/image.jpg', text_prompt='person, car, building')]

IMPORTANT: Execute the tool call exactly as provided, do not modify it, and do not execute it multiple times. The tool can handle multiple objects in a single call.

""",
            tools=[self.grounded_sam_tool],
            enable_session_persistence=False,
        )
    
    def _create_analysis_agent(self) -> Agent:
        """Create agent specialized for comprehensive analysis."""
        return Agent(
            self.client,
            model="llama4:scout",
            instructions="""
You are a comprehensive image analysis assistant. Your job is to provide detailed analysis combining segmentation and exposure results.

You will receive multiple images in the message:
1. The original image (first image)
2. Segmented object images (next few images)
3. Exposure heatmaps (remaining images)

Your task is to:
1. Analyze the original image to understand the overall scene
2. Examine each segmented image to identify what objects were found and their characteristics
3. Study each exposure heatmap to understand the exposure quality of different areas
4. Provide a comprehensive analysis including:
   - Summary of objects found and their characteristics
   - Analysis of exposure quality for each object/area
   - Overall assessment and recommendations
   - Professional insights about the image quality

Be thorough and professional in your analysis. Use the visual information from all provided images to make informed conclusions. Reference specific details you can see in the images.
""",
            tools=[],  # No tools needed for analysis
            enable_session_persistence=False,
        )
    
    def analyze_image(self, user_query: str, image_path: str, use_openai: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive image analysis using multi-turn workflow.
        
        Args:
            user_query: User's analysis request
            image_path: Path to the image file
            use_openai: Whether to use OpenAI for comprehensive analysis (default: False, uses Llama Stack)
            
        Returns:
            Dict containing analysis results and metadata
        """
        # Validate image path
        if not os.path.exists(image_path):
            return {
                "error": f"Image file not found: {image_path}",
                "success": False
            }
        
        print(f"\n{'='*80}")
        print(f"Starting Multi-Turn Image Analysis Workflow")
        print(f"Query: {user_query}")
        print(f"Image: {image_path}")
        print(f"{'='*80}")
        
        try:
            # STEP 1: Analysis and Tool Call Generation
            print("\n🔄 STEP 1: Analysis and Tool Call Generation")
            print("-" * 50)
            
            segmentation_analysis_result = self._perform_analysis(user_query, image_path)
            if not segmentation_analysis_result.get("success", False):
                return segmentation_analysis_result
            
            full_response = segmentation_analysis_result.get("full_response", "")
            if not full_response:
                return {
                    "error": "No response generated from analysis",
                    "user_query": user_query,
                    "image_path": image_path,
                    "success": False
                }
            
            # STEP 2: Execute Segmentation Tool
            print("\n🔄 STEP 2: Execute Segmentation Tool")
            print("-" * 50)
            
            segmentation_results = self._execute_segmentation_tool(full_response, image_path)
            if not segmentation_results.get("success", False):
                return segmentation_results
            
            segmented_images = segmentation_results.get("segmented_images", [])
            if not segmented_images:
                return {
                    "error": "No objects were segmented from the image",
                    "user_query": user_query,
                    "image_path": image_path,
                    "success": False
                }
            
            # STEP 3: Exposure Evaluation for each segmented object
            print("\n🔄 STEP 3: Exposure Evaluation")
            print("-" * 50)
            
            exposure_results = []
            for img_info in segmented_images:
                segmented_image_path = img_info.get("image_path")
                if os.path.exists(segmented_image_path):
                    print(f"Evaluating exposure for: {img_info.get('phrase', 'object')}")
                    exp_result = self._perform_exposure_evaluation(segmented_image_path)
                    if exp_result.get("success", False):
                        exposure_results.append({
                            "object_info": img_info,
                            "exposure_data": exp_result
                        })

            # STEP 3.5: Rule RAG and Dual-Grounding
            print("\n馃攧 STEP 3.5: Rule RAG and Dual-Grounding")
            print("-" * 50)

            grounding_seed_records = self._build_grounding_seed_records(exposure_results)
            retrieved_rules = self.rule_rag.retrieve(
                user_query=user_query,
                grounding_records=grounding_seed_records,
                top_k=5
            )
            dual_grounding_manifest = self.dual_grounding_builder.build(
                segmentation_results=segmentation_results,
                exposure_results=exposure_results,
                retrieved_rules=retrieved_rules
            )
            manifest_base = os.path.splitext(os.path.basename(image_path))[0]
            manifest_dir = segmentation_results.get("output_directory") or os.path.dirname(image_path)
            dual_grounding_manifest_path = self.dual_grounding_builder.save_manifest(
                dual_grounding_manifest,
                manifest_dir,
                manifest_base
            )
            print(f"Retrieved {len(retrieved_rules)} exposure rules")
            print(f"Dual-grounding manifest saved to: {dual_grounding_manifest_path}")
            
            # STEP 4: Comprehensive Analysis
            print("\n🔄 STEP 4: Comprehensive Analysis")
            if use_openai:
                print("Using OpenAI API for comprehensive analysis")
            else:
                print("Using Llama Stack for comprehensive analysis")
            print("-" * 50)
            
            analysis_result = self._perform_comprehensive_analysis(
                user_query,
                image_path,
                segmentation_results,
                exposure_results,
                retrieved_rules=retrieved_rules,
                dual_grounding_manifest=dual_grounding_manifest,
                use_openai=use_openai
            )
            
            # Combine all results
            final_results = {
                "user_query": user_query,
                "image_path": image_path,
                "analysis_result": segmentation_analysis_result,
                "segmentation_results": segmentation_results,
                "exposure_results": exposure_results,
                "rule_rag_results": retrieved_rules,
                "dual_grounding_manifest": dual_grounding_manifest,
                "dual_grounding_manifest_path": dual_grounding_manifest_path,
                "comprehensive_analysis_result": analysis_result,
                "success": True
            }
            
            return final_results
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "user_query": user_query,
                "image_path": image_path,
                "success": False
            }

    def _build_grounding_seed_records(self, exposure_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten segmentation and exposure outputs for rule retrieval."""
        records = []
        for item in exposure_results:
            object_info = item.get("object_info", {})
            exposure_data = item.get("exposure_data", {})
            records.append({
                "label": object_info.get("phrase", "object"),
                "box": object_info.get("box"),
                "mask_path": object_info.get("image_path"),
                "heatmap_path": exposure_data.get("heatmap_path"),
                "average_exposure_score": exposure_data.get("average_exposure_score"),
                "min_exposure_score": exposure_data.get("min_exposure_score"),
                "max_exposure_score": exposure_data.get("max_exposure_score"),
                "exposure_std": exposure_data.get("exposure_std"),
            })
        return records
    
    def _perform_analysis(self, user_query: str, image_path: str) -> Dict[str, Any]:
        """Perform analysis and generate tool call."""
        session_id = self.segmentation_agent.create_session(f"analysis-{uuid.uuid4()}")
        
        # Prepare message content with image and text
        message_content = []
        
        # Add original image so the agent can "see" it
        try:
            original_image_data = encode_image_to_base64(image_path)
            message_content.append({
                "type": "image", 
                "image": {"data": original_image_data}
            })
            print(f"✅ Added original image to analysis agent: {image_path}")
        except Exception as e:
            print(f"❌ Failed to add original image to analysis agent: {e}")
            return {"success": False, "error": f"Failed to encode image: {e}"}
        
        # Add text content
        message_content.append({
            "type": "text", 
            "text": f"Analyze this image based on the request: '{user_query}'. Identify the objects that need to be segmented for this analysis and create a text prompt with specific object names (not questions). Use image_path='{image_path}' in your tool call."
        })
        
        message = {
            "role": "user",
            "content": message_content
        }
        
        response = self.segmentation_agent.create_turn(
            messages=[message],
            session_id=session_id,
        )
        
        # Extract the full response content
        print("\nAnalysis Agent Execution Steps:")
        print("-" * 50)
        
        logs = list(EventLogger().log(response))
        full_response = ""
        
        for log in logs:
            log.print()
            
            # Collect the full response content from inference logs
            if hasattr(log, "role") and log.role == "inference" and hasattr(log, "content"):
                full_response += log.content + "\n"
            # Also check for content in other log types
            elif hasattr(log, "content") and log.content:
                full_response += log.content + "\n"
        
        if full_response:
            print(f"✅ Successfully extracted analysis response: {len(full_response)} characters")
            return {
                "success": True,
                "full_response": full_response
            }
        else:
            print("❌ No content found in response")
            return {"success": False, "error": "No response generated"}
    
    def _execute_segmentation_tool(self, full_response: str, image_path: str) -> Dict[str, Any]:
        """Execute segmentation tool using the full response from analysis agent."""
        session_id = self.segmentation_executor_agent.create_session(f"executor-{uuid.uuid4()}")
        
        # Send the full response from analysis agent to the executor agent
        # Also provide the correct image path to ensure accuracy
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Execute this tool call exactly once:\n\n{full_response}\n\nThe correct image path is: {image_path}"}
            ]
        }
        
        response = self.segmentation_executor_agent.create_turn(
            messages=[message],
            session_id=session_id,
        )
        
        # Extract segmentation results
        print("\nTool Execution Steps:")
        print("-" * 50)
        
        logs = list(EventLogger().log(response))
        for log in logs:
            log.print()
            
            if getattr(log, "role", None) == "tool_execution" and hasattr(log, "content"):
                content = log.content
                if "Response:" in content:
                    json_str = content.split("Response:")[-1].strip()
                    try:
                        tool_result = json.loads(json_str)
                        if "segmented_images" in tool_result:
                            return {
                                "success": True,
                                "segmented_images": tool_result.get("segmented_images", []),
                                "total_masks": tool_result.get("total_masks", 0),
                                "text_prompt": tool_result.get("text_prompt", ""),
                                "output_directory": tool_result.get("output_directory", "")
                            }
                    except Exception as e:
                        print(f"Failed to parse segmentation result: {e}")
        
        return {"success": False, "error": "Segmentation execution failed"}
    
    def _perform_exposure_evaluation(self, segmented_image_path: str) -> Dict[str, Any]:
        """Perform exposure evaluation on a segmented image."""
        session_id = self.exposure_agent.create_session(f"exposure-{uuid.uuid4()}")
        
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Evaluate the exposure of this segmented image: {segmented_image_path}"}
            ]
        }
        
        response = self.exposure_agent.create_turn(
            messages=[message],
            session_id=session_id,
        )
        
        # Extract exposure results
        print(f"\nEvaluating exposure for segmented image: {segmented_image_path}")
        
        logs = list(EventLogger().log(response))
        for log in logs:
            log.print()
            
            if getattr(log, "role", None) == "tool_execution" and hasattr(log, "content"):
                content = log.content
                if "Response:" in content:
                    json_str = content.split("Response:")[-1].strip()
                    try:
                        tool_result = json.loads(json_str)
                        if "average_exposure_score" in tool_result:
                            return {
                                "success": True,
                                "average_exposure_score": tool_result.get("average_exposure_score"),
                                "min_exposure_score": tool_result.get("min_exposure_score"),
                                "max_exposure_score": tool_result.get("max_exposure_score"),
                                "exposure_std": tool_result.get("exposure_std"),
                                "heatmap_path": tool_result.get("heatmap_path"),
                                "heatmap_filename": tool_result.get("heatmap_filename")
                            }
                    except Exception as e:
                        print(f"Failed to parse exposure result: {e}")
        
        return {"success": False, "error": "Exposure evaluation failed"}
    
    def _perform_comprehensive_analysis_with_openai(self, user_query: str, image_path: str,
                                                   segmentation_results: Dict, exposure_results: List,
                                                   retrieved_rules: Optional[List[Dict[str, Any]]] = None,
                                                   dual_grounding_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis using OpenAI API."""
        if not self.openai_client:
            return {
                "success": False,
                "error": "OpenAI client not initialized. Please provide API key."
            }
        
        print(f"Using OpenAI model: {self.openai_model}")
        
        # Clean text function to handle special Unicode characters
        def clean_text(text):
            """Clean text by removing/replacing problematic characters."""
            if not isinstance(text, str):
                text = str(text)
            # Remove special Unicode line/paragraph separators and other invisible characters
            text = text.replace('\u2028', ' ')  # Line Separator
            text = text.replace('\u2029', ' ')  # Paragraph Separator
            text = text.replace('\u200b', '')   # Zero Width Space
            text = text.replace('\u200c', '')   # Zero Width Non-Joiner
            text = text.replace('\u200d', '')   # Zero Width Joiner
            text = text.replace('\ufeff', '')   # Zero Width No-Break Space
            return text
        
        # Clean user query
        user_query = clean_text(user_query)
        
        # Prepare message content with images
        message_content = []
        
        # Add original image
        try:
            original_image_data = encode_image_to_base64(image_path)
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{original_image_data}"
                }
            })
            print(f"✅ Added original image to OpenAI analysis: {image_path}")
        except Exception as e:
            print(f"❌ Failed to add original image: {e}")
        
        # Add segmented images
        segmented_images = segmentation_results.get("segmented_images", [])
        for i, img_info in enumerate(segmented_images):
            segmented_image_path = img_info.get("image_path")
            if segmented_image_path and os.path.exists(segmented_image_path):
                try:
                    segmented_image_data = encode_image_to_base64(segmented_image_path)
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{segmented_image_data}"
                        }
                    })
                    # Clean phrase text for display
                    phrase = clean_text(img_info.get('phrase', 'object'))
                    print(f"✅ Added segmented image {i+1} to OpenAI analysis: {phrase}")
                except Exception as e:
                    print(f"❌ Failed to add segmented image {i+1}: {e}")
        
        # Add exposure heatmaps
        for i, exp_data in enumerate(exposure_results):
            exp_info = exp_data.get("exposure_data", {})
            heatmap_path = exp_info.get("heatmap_path")
            if heatmap_path and os.path.exists(heatmap_path):
                try:
                    heatmap_data = encode_image_to_base64(heatmap_path)
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{heatmap_data}"
                        }
                    })
                    print(f"✅ Added exposure heatmap {i+1} to OpenAI analysis")
                except Exception as e:
                    print(f"❌ Failed to add exposure heatmap {i+1}: {e}")
        
        # Prepare analysis text (clean all text data before formatting)
        text_prompt = clean_text(str(segmentation_results.get('text_prompt', '')))
        
        rule_context = self.rule_rag.format_rules_for_prompt(retrieved_rules or [])
        grounding_context = self.dual_grounding_builder.format_for_prompt(dual_grounding_manifest or {})

        analysis_text = f"""
User Query: {user_query}
Please provide a comprehensive analysis and reply to the user query of the image based on the following information:

Images provided:
1. Original image (first image)
2. Segmented object images (next {len(segmented_images)} images)
3. Exposure heatmaps (remaining images)

Segmentation Results:
- Total objects found: {segmentation_results.get('total_masks', 0)}
- Objects segmented: {len(segmented_images)}
- Text prompt used: {text_prompt}

Exposure Analysis Results:
- Number of objects analyzed: {len(exposure_results)}

Rule RAG:
{rule_context}

Dual-Grounding:
{grounding_context}

Please provide a detailed analysis including:
1. Summary of what objects were found and their characteristics (based on the segmented images)
2. Analysis of exposure quality for each object (based on the heatmaps)
3. Rule-grounded reasoning: cite the retrieved rule IDs that justify your claims
4. Dual-grounded evidence: refer to region ids, masks, boxes, heatmaps, and exposure scores
5. Professional insights, overall assessment and recommendations to improve the image quality and exposure level

Be thorough and professional in your analysis. Use the visual information from the provided images to make informed conclusions.
Do not invent objects that are not in the dual-grounding evidence.
"""
        
        # Add text content
        message_content.append({
            "type": "text",
            "text": analysis_text
        })
        
        try:
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                max_tokens=4096
            )
            
            analysis_content = response.choices[0].message.content
            
            # Clean the response content 
            # Remove problematic characters that might cause encoding issues
            if analysis_content:
                analysis_content = clean_text(analysis_content)
                # Additional cleaning 
                analysis_content = analysis_content.replace("\n\n", " ").replace("**", "")
            
            print("\nOpenAI Analysis Results:")
            print("-" * 50)
            print(analysis_content)
            
            return {
                "success": True,
                "analysis": analysis_content,
                "model_used": self.openai_model,
                "provider": "openai"
            }
            
        except Exception as e:
            print(f"❌ OpenAI API call failed: {e}")
            return {
                "success": False,
                "error": f"OpenAI API call failed: {str(e)}"
            }
    
    def _perform_comprehensive_analysis(self, user_query: str, image_path: str, 
                                      segmentation_results: Dict, exposure_results: List,
                                      retrieved_rules: Optional[List[Dict[str, Any]]] = None,
                                      dual_grounding_manifest: Optional[Dict[str, Any]] = None,
                                      use_openai: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive analysis combining all results.
        
        Args:
            user_query: User's analysis request
            image_path: Path to original image
            segmentation_results: Results from segmentation
            exposure_results: Results from exposure evaluation
            retrieved_rules: Rule RAG results
            dual_grounding_manifest: Visual-textual grounding evidence
            use_openai: Whether to use OpenAI API instead of Llama Stack
        
        Returns:
            Dict containing analysis results
        """
        # Use OpenAI if requested
        if use_openai:
            return self._perform_comprehensive_analysis_with_openai(
                user_query,
                image_path,
                segmentation_results,
                exposure_results,
                retrieved_rules=retrieved_rules,
                dual_grounding_manifest=dual_grounding_manifest
            )
        
        # Otherwise use Llama Stack
        session_id = self.analysis_agent.create_session(f"analysis-{uuid.uuid4()}")
        
        # Prepare message content with images and text
        message_content = []
        
        # Add original image
        try:
            original_image_data = encode_image_to_base64(image_path)
            message_content.append({
                "type": "image", 
                "image": {"data": original_image_data}
            })
            print(f"✅ Added original image to analysis: {image_path}")
        except Exception as e:
            print(f"❌ Failed to add original image: {e}")
        
        # Add segmented images
        segmented_images = segmentation_results.get("segmented_images", [])
        for i, img_info in enumerate(segmented_images):
            segmented_image_path = img_info.get("image_path")
            if segmented_image_path and os.path.exists(segmented_image_path):
                try:
                    segmented_image_data = encode_image_to_base64(segmented_image_path)
                    message_content.append({
                        "type": "image", 
                        "image": {"data": segmented_image_data}
                    })
                    print(f"✅ Added segmented image {i+1} to analysis: {img_info.get('phrase', 'object')}")
                except Exception as e:
                    print(f"❌ Failed to add segmented image {i+1}: {e}")
        
        # Add exposure heatmaps
        for i, exp_data in enumerate(exposure_results):
            exp_info = exp_data.get("exposure_data", {})
            heatmap_path = exp_info.get("heatmap_path")
            if heatmap_path and os.path.exists(heatmap_path):
                try:
                    heatmap_data = encode_image_to_base64(heatmap_path)
                    message_content.append({
                        "type": "image", 
                        "image": {"data": heatmap_data}
                    })
                    print(f"✅ Added exposure heatmap {i+1} to analysis")
                except Exception as e:
                    print(f"❌ Failed to add exposure heatmap {i+1}: {e}")
        
        # Prepare analysis text
        rule_context = self.rule_rag.format_rules_for_prompt(retrieved_rules or [])
        grounding_context = self.dual_grounding_builder.format_for_prompt(dual_grounding_manifest or {})

        analysis_text = f"""
User Query: {user_query}
Please provide a comprehensive analysis and reply the user queryof the image based on the following information:

Images provided:
1. Original image (first image)
2. Segmented object images (next {len(segmented_images)} images)
3. Exposure heatmaps (remaining images)

Segmentation Results:
- Total objects found: {segmentation_results.get('total_masks', 0)}
- Objects segmented: {len(segmented_images)}

Exposure Analysis Results:
- Number of objects analyzed: {len(exposure_results)}

Rule RAG:
{rule_context}

Dual-Grounding:
{grounding_context}

Please provide a detailed analysis including:
1. Summary of what objects were found and their characteristics (based on the segmented images)
2. Analysis of exposure quality for each object (based on the heatmaps)
3. Rule-grounded reasoning: cite the retrieved rule IDs that justify your claims
4. Dual-grounded evidence: refer to region ids, masks, boxes, heatmaps, and exposure scores
5. Professional insights, overall assessment and recommendations to improve the image quality and exposure level

Be thorough and professional in your analysis. Use the visual information from the provided images to make informed conclusions.
Do not invent objects that are not in the dual-grounding evidence.
"""
        
        # Add text content
        message_content.append({"type": "text", "text": analysis_text})
        
        message = {
            "role": "user",
            "content": message_content
        }
        
        response = self.analysis_agent.create_turn(
            messages=[message],
            session_id=session_id,
        )
        
        print("\nComprehensive Analysis Results:")
        print("-" * 50)
        
        logs = list(EventLogger().log(response))
        analysis_content = ""
        
        for log in logs:
            log.print()
            
            # LogEvent objects have a 'content' attribute
            # Skip the "Assistant> " prefix (color=cyan, empty content or just prefix)
            # Capture the actual yellow-colored content (the analysis text)
            if hasattr(log, "content"):
                content = log.content
                # Skip empty content and the "Assistant> " prefix
                if content and content.strip() and content.strip() != "Assistant>":
                    analysis_content += content
        
        # If we captured content from logs, return it
        if analysis_content.strip():
            print(f"\n✅ Captured Llama Stack analysis: {len(analysis_content)} characters")
            return {
                "success": True,
                "analysis": analysis_content.strip(),
                "model_used": "llama4:scout",
                "provider": "llama_stack"
            }
        
        # Fallback: check output_message attribute
        if hasattr(response, 'output_message') and response.output_message:
            print(f"\n✅ Got Llama analysis from output_message")
            return {
                "success": True,
                "analysis": response.output_message.content,
                "model_used": "llama4:scout",
                "provider": "llama_stack"
            }
        
        print(f"\n❌ Failed to capture Llama analysis - no content found in {len(logs)} log events")
        return {"success": False, "error": "Analysis failed - no content captured"}
    
    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive report from the analysis results."""
        if not analysis_results.get("success", False):
            return f"Analysis failed: {analysis_results.get('error', 'Unknown error')}"
        
        report = []
        report.append("# Multi-Turn Image Analysis Report")
        report.append("")
        
        # User query
        report.append(f"**User Query:** {analysis_results.get('user_query', 'N/A')}")
        report.append(f"**Image Path:** {analysis_results.get('image_path', 'N/A')}")
        report.append("")
        
        # Analysis results
        analysis_result = analysis_results.get("analysis_result", {})
        if analysis_result.get("success", False) and analysis_result.get("full_response"):
            report.append("## Analysis and Tool Call Generation")
            report.append(f"**Analysis Response:** {analysis_result.get('full_response', 'N/A')[:200]}...")
        else:
            report.append("## Analysis and Tool Call Generation")
            report.append("Analysis failed or no response generated.")
        
        # Segmentation results
        seg_results = analysis_results.get("segmentation_results", {})
        if seg_results.get("success", False):
            report.append("\n## Object Segmentation Results")
            report.append(f"**Total Objects Segmented:** {seg_results.get('total_masks', 0)}")
            report.append(f"**Text Prompt Used:** {seg_results.get('text_prompt', 'N/A')}")
            
            for i, img_info in enumerate(seg_results.get("segmented_images", [])):
                report.append(f"\n### Object {i+1}")
                report.append(f"- **Description:** {img_info.get('phrase', 'N/A')}")
                report.append(f"- **Confidence Score:** {img_info.get('score', 'N/A'):.4f}")
                report.append(f"- **Segmented Image:** {img_info.get('filename', 'N/A')}")
        else:
            report.append("\n## Object Segmentation Results")
            report.append("Segmentation failed or no objects found.")
        
        # Exposure results
        exp_results = analysis_results.get("exposure_results", [])
        if exp_results:
            report.append("\n## Exposure Analysis Results")
            
            for i, exp_data in enumerate(exp_results):
                obj_info = exp_data.get("object_info", {})
                exp_info = exp_data.get("exposure_data", {})
                
                report.append(f"\n### Exposure Analysis {i+1}")
                report.append(f"- **Object:** {obj_info.get('phrase', 'N/A')}")
                report.append(f"- **Average Score:** {exp_info.get('average_exposure_score', 'N/A')}")
                report.append(f"- **Min Score:** {exp_info.get('min_exposure_score', 'N/A')}")
                report.append(f"- **Max Score:** {exp_info.get('max_exposure_score', 'N/A')}")
                report.append(f"- **Standard Deviation:** {exp_info.get('exposure_std', 'N/A')}")
                report.append(f"- **Heatmap:** {exp_info.get('heatmap_filename', 'N/A')}")
        else:
            report.append("\n## Exposure Analysis Results")
            report.append("No exposure analysis was performed.")

        # Rule RAG and Dual-Grounding
        rule_results = analysis_results.get("rule_rag_results", [])
        if rule_results:
            report.append("\n## Rule RAG Results")
            for rule in rule_results:
                report.append(
                    f"- **{rule.get('rule_id')} {rule.get('title')}** "
                    f"(score={rule.get('retrieval_score')}): {rule.get('guidance')}"
                )

        dual_grounding_manifest = analysis_results.get("dual_grounding_manifest", {})
        if dual_grounding_manifest:
            report.append("\n## Dual-Grounding Results")
            manifest_path = analysis_results.get("dual_grounding_manifest_path")
            if manifest_path:
                report.append(f"**Manifest:** {manifest_path}")
            for region in dual_grounding_manifest.get("regions", []):
                report.append(
                    f"- **{region.get('region_id')}** {region.get('label')} | "
                    f"quality={region.get('exposure_quality')} | "
                    f"avg={region.get('average_exposure_score')} | "
                    f"rules={', '.join(region.get('rule_ids') or [])}"
                )
        
        # Final analysis
        comprehensive_analysis = analysis_results.get("comprehensive_analysis_result", {})
        if comprehensive_analysis.get("success", False):
            report.append("\n## Comprehensive Analysis")
            
            # Add model information
            provider = comprehensive_analysis.get("provider", "unknown")
            model_used = comprehensive_analysis.get("model_used", "unknown")
            report.append(f"**Analysis Provider:** {provider}")
            report.append(f"**Model Used:** {model_used}")
            report.append("")
            
            report.append(comprehensive_analysis.get("analysis", "No analysis provided."))
        else:
            report.append("\n## Comprehensive Analysis")
            report.append("Comprehensive analysis failed or no analysis provided.")
        
        return "\n".join(report)


def main():
    """Test the improved workflow."""
    workflow = ImageAnalysisWorkflow()
    
    # Test case
    query = "Please analyze the exposure level of this image"
    image_path = "/path/to/your/image.jpg"
    
    print("Testing improved multi-turn workflow...")
    results = workflow.analyze_image(query, image_path)
    
    if results.get("success", False):
        print("\n✅ Analysis completed successfully!")
        print("\n" + "="*80)
        print("IMPROVED WORKFLOW RESULTS")
        print("="*80)
        print(workflow.generate_comprehensive_report(results))
    else:
        print(f"\n❌ Analysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main() 
