from pathlib import Path
import textwrap
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent  # directory of the current file
EXP_FILE_DIR = Path(__file__).resolve().parents[2] / "experiments"

OpenRouter_Image_Models = [
    "google/gemini-2.5-flash-lite",  # hundreds of B (lightweight Gemini) $0.1
    # "meta-llama/llama-4-maverick",  # ~17B $0.15
    # "google/gemma-3-4b-it",  # $0.02
    # "mistralai/mistral-small-3.2-24b-instruct",  # ~24B $0.02
    # "qwen/qwen2.5-vl-32b-instruct",  # $0.02
    # "microsoft/phi-4-multimodal-instruct",  # $0.05
    # "openai/gpt-5-mini",  # 0.25
    # "openai/gpt-4o-2024-08-06",  # est. ~200B–1T $2.5
    # "anthropic/claude-3.7-sonnet",  # ~150–250B $3.0
    # excluded the below
    # "openai/gpt-5",  # 1.25
    # "qwen/qwen-vl-plus",  # ~235B total, ~22B activated $0.21
    # "openai/gpt-4.1-mini",  # ~8–10B (GPT-4 mini variant) $0.4
    # "openai/gpt-oss-20b",  # $0.04
]

OpenRouter_Text_Models = [
    "google/gemini-2.5-flash-lite",  # hundreds of B (lightweight Gemini) $0.1
    "meta-llama/llama-4-maverick",  # ~17B $0.15
    "google/gemma-3-4b-it",  # $0.02
    "mistralai/mistral-small-3.2-24b-instruct",  # ~24B $0.02
    "qwen/qwen2.5-vl-32b-instruct",  # $0.02
    "microsoft/phi-4-multimodal-instruct",  # $0.05
    "openai/gpt-5-mini",  # 0.25
    "openai/gpt-4o-2024-08-06",  # est. ~200B–1T $2.5
    "anthropic/claude-3.7-sonnet",  # ~150–250B $3.0
    # "meta-llama/llama-3.1-8b-instruct",  # parameter count: 8B
    # "meta-llama/llama-4-maverick",  # parameter count: 17B
    # "qwen/qwen3-32b",  # parameter count: 32B
    # "openai/gpt-3.5-turbo",  # parameter count: 20-40B
    # "meta-llama/llama-3.3-70b-instruct",  # parameter count: 70B
    # "openai/gpt-3.5-turbo",  # parameter count: 20-40B
    # "meta-llama/llama-3.3-70b-instruct",  # parameter count: 70B
    # "anthropic/claude-3.7-sonnet",  # parameter count: 150-250B
    # "google/gemini-2.5-pro",  # parameter count: 25B
]

Groq_Models = [
    "llama-3.1-8b-instant",
    "mistral-saba-24b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

OpenAI_Models = [
    "gpt-3.5-turbo",
]

unsloth_models = ["marcelbinz/Llama-3.1-Minitaur-8B-adapter"]


class CommonConfig:
    llm_provider = "OpenRouter"  # "groq"  #      #
    DEFAULT_MODELS = OpenRouter_Image_Models
    temperature = 0.3
    max_tokens = 25
    num_samples = 50
    accumulate_context = True
    context_window = 10  # keep last 100 interactions
    block_size = 10  # number of samples to process in one batch
    experiment_function = "run_experiment"
    system_prompt = "You are a helpful assistant."
    extra_params = {}
    true_column = "ground_truth"
    pred_column = "prediction"
    input_column = "input_values"
    exp_id = "040825_text_image_remapping_2"
    reverse_order = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class SubtitleDuration(CommonConfig):
    experiment_type = "text"
    experiment_name = "subtitle_duration"
    experiment_module = (
        f"experiments.{experiment_type}.{experiment_name}.experiment_run"
    )

    user_prompt_zero_shot = textwrap.dedent(
        """
            Estimate how many seconds it takes to say out loud the following text:
            {sentence}
            Final Answer:
        """
    )

    def __init__(
        self,
        experiment_mode="text",
        audio_mode="normal",
        system_prompt="You are a speech duration estimator. Do not reason. Output only the final answer as one number after Final Answer:",
        verbal_steer=False,
        numerical_steer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experiment_mode = experiment_mode
        self.multi_modal = self.experiment_mode != "text"
        self.audio_mode = audio_mode
        self.verbal_steer = verbal_steer
        self.numerical_steer = numerical_steer
        self.system_prompt = system_prompt

        self.user_prompt_zero_shot = self.user_prompt_zero_shot
        self.user_prompt_zero_shot = textwrap.dedent(self.user_prompt_zero_shot)

        self.experiment_module = (
            f"experiments.{self.experiment_type}.{self.experiment_name}.experiment_run"
        )
        self.experiment_path = (
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/runs/{self.exp_id}/"
        )
        self.experiment_files = [
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/AMI_Corpus/experiment_files/1.0_8.0_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/AMI_Corpus/experiment_files/5.0_20.0_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/AMI_Corpus/experiment_files/15.0_50.0_100samples.csv",
        ]


class AudioSpeechDuration(CommonConfig):
    experiment_type = "text_audio"
    experiment_name = "speech_duration"
    experiment_module = (
        f"experiments.{experiment_type}.{experiment_name}.experiment_run"
    )
    base_1_text = """
        You are given a transcript of a speech: 
        
        {text_representation}
        """

    base_1_audio = """
        You are given an audio recording of a speech.
        """

    base_1_text_audio = """
        You are given both an audio recording and a transcript of a speech.
        """

    base_2 = """ 
        In both, a sentence is read out loud.

        Here is the transcript:
        {text_representation}
        """

    base_3 = """ 
        Estimate how many seconds it will take to read the sentence out loud.

        Do not explain or reason. Only output the final answer as a number:
        
        Final Answer:
        """

    # Abalation prompts

    option_1_vs = """
        The given data is noisy and may contain artifacts. You should behave like a Bayesian observer and take into account prior and likelihood in your predictions.     
    """

    option_1_ns = """
        The given data is noisy and may contain artifacts. The prior ground truth will be in the range of {value_range}.     
    """

    def __init__(
        self,
        experiment_mode="text_audio",
        audio_mode="normal",
        system_prompt="You are a speech duration estimator.",
        verbal_steer=False,
        numerical_steer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experiment_mode = experiment_mode
        self.multi_modal = self.experiment_mode != "text"
        self.audio_mode = audio_mode
        self.verbal_steer = verbal_steer
        self.numerical_steer = numerical_steer
        self.system_prompt = system_prompt

        if self.experiment_mode == "text_audio":
            self.user_prompt_zero_shot = self.base_1_text_audio
        if self.experiment_mode == "text":
            self.user_prompt_zero_shot = self.base_1_text
        if self.experiment_mode == "audio":
            self.user_prompt_zero_shot = self.base_1_audio

        if self.verbal_steer:
            self.user_prompt_zero_shot += self.option_1_vs
        if self.numerical_steer:
            self.user_prompt_zero_shot += self.option_1_ns

        if self.experiment_mode == "text_audio":
            self.user_prompt_zero_shot += self.base_2

        self.user_prompt_zero_shot += self.base_3
        self.user_prompt_zero_shot = textwrap.dedent(self.user_prompt_zero_shot)

        self.experiment_module = (
            f"experiments.{self.experiment_type}.{self.experiment_name}.experiment_run"
        )
        self.experiment_path = (
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/runs/{self.exp_id}/"
        )
        self.experiment_files = [
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/3.0_6.0/speech_duration_3.0_6.0_16samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/4.5_7.5/speech_duration_4.5_7.5_24samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/6.0_9.0/speech_duration_6.0_9.0_19samples.csv",
        ]


class ImageMarkerLocation(CommonConfig):
    experiment_type = "text_image"
    experiment_name = "marker_location"

    # experiment specific parameters - manually set
    ascii_line_choice = "ascii_line"  # "high_noise_ascii_line"
    text_shift_flag = False
    remapping_test = False  # True for remapping test, False for regular marker location
    remapping_test_phase = (
        1  # 1 for first phase, 2 for second phase, to compare the impact of remapping
    )

    process_factor = 0.0

    # Construct prompts here - template prompts

    base_1_text_system = """
        You are given a text-based representation of a horizontal line where a red dot is marked as ""O""

        The left end of the line corresponds to 0.0, the right end corresponds to 1.0.

        Estimate the horizontal position of the **center** of the red dot as a decimal number between 0 and 1.
        """

    base_1_image_system = """
        You are given an image showing a red dot on a horizontal line. 

        The left end of the line corresponds to 0.0, the right end corresponds to 1.0.

        Estimate the horizontal position of the **center** of the red dot as a decimal number between 0 and 1.
        """

    base_1_text_image_system = """
        You are given both an image and a text-based description of a horizontal line.

        In both, a red dot appears on a horizontal line

        The left end of the line corresponds to 0.0, the right end corresponds to 1.0.

        Estimate the horizontal position of the **center** of the red dot as a decimal number between 0 and 1.
        """

    base_3_system = """ 
        Do not explain or reason. Only output the final answer as a number after Final Answer:
        """

    base_1_text = """
        Here is the text-based representation of the line with a red dot marked as ""O"":
        {text_representation}
        """

    base_1_image = """
        Consider the image provided.
        """

    base_1_text_image = """
        Consider the image provided and the text-based representation of the line, where the red dot is marked as ""O"".
        {text_representation}
        """

    base_3 = """ 
        Final Answer:
        """

    # Abalation prompts

    option_1_vs_system = """
        The given data is noisy and may contain artifacts. You should behave like a Bayesian observer and take into account prior and likelihood in your predictions.     
    """

    option_1_ns_system = """
        The given data is noisy and may contain artifacts. For 10 previous observations, the values were observed to lie in the range of {value_range}.     
    """

    option_1_ps_system = """
        The given data is noisy and may contain artifacts. For 10 previous observations, the values were observed to lie in the range of {value_range}.     
    """

    def __init__(
        self,
        experiment_mode="text_image",
        image_mode="normal",
        system_prompt="You are a marker location estimator.",
        verbal_steer=False,
        numerical_steer=False,
        prior_steer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experiment_mode = experiment_mode
        self.multi_modal = self.experiment_mode != "text"
        self.image_mode = image_mode
        self.verbal_steer = verbal_steer
        self.numerical_steer = numerical_steer
        self.prior_steer = prior_steer
        self.system_prompt = system_prompt

        if self.experiment_mode == "text_image":
            self.system_prompt += self.base_1_text_image_system
            self.user_prompt_zero_shot = self.base_1_text_image
        if self.experiment_mode == "text":
            self.system_prompt += self.base_1_text_system
            self.user_prompt_zero_shot = self.base_1_text
        if self.experiment_mode == "image":
            self.system_prompt += self.base_1_image_system
            self.user_prompt_zero_shot = self.base_1_image

        if self.verbal_steer:
            self.system_prompt += self.option_1_vs_system
        if self.numerical_steer:
            self.system_prompt += self.option_1_ns_system
        if self.prior_steer:
            self.system_prompt += self.option_1_ps_system

        self.system_prompt += self.base_3_system
        self.user_prompt_zero_shot += self.base_3
        self.system_prompt = textwrap.dedent(self.system_prompt)
        self.user_prompt_zero_shot = textwrap.dedent(self.user_prompt_zero_shot)

        self.experiment_module = (
            f"experiments.{self.experiment_type}.{self.experiment_name}.experiment_run"
        )
        self.experiment_path = (
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/runs/{self.exp_id}/"
        )
        self.experiment_files = [
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/0.1_0.5/marker_loc_0.1_0.5_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/0.3_0.8/marker_loc_0.3_0.8_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/0.5_0.9/marker_loc_0.5_0.9_100samples.csv",
        ]


class ImageLineLengthRatio(CommonConfig):
    experiment_type = "text_image"
    experiment_name = "line_length_ratio"

    # experiment specific parameters
    ascii_line_choice = "ascii_line"  # "high_noise_ascii_line"

    # Construct prompts here - template prompts

    base_1_text_system = """
        You are given a text-based representation of two lines. The representation is delimited by vertical bars: |...|. Ignore space characters when considering its length.

        Estimate the ratio of the shorter line to the longer line as a decimal number between 0 and 1.
        """

    base_1_image_system = """
        You are given an image of two lines.

        Estimate the ratio of the shorter line to the longer line as a decimal number between 0 and 1.
        """

    base_1_text_image_system = """
        You are given both an image and a text-based representation of two lines. 

        The text representation is delimited by vertical bars: |...|. Ignore space characters when considering its length.

        Estimate the ratio of the shorter line to the longer line as a decimal number between 0 and 1.
        """

    base_3_system = """ 
        Do not explain or reason. Only output the final answer as a number after Final Answer:
        """

    base_1_text = """
        Here is the text-based representation of two lines.

        {text_representation_1}
        {text_representation_2} 
        """

    base_1_image = """
        Consider the image provided.
        """

    base_1_text_image = """
        Consider the image provided and the text-based representation of two lines. 
        {text_representation_1}
        {text_representation_2} 
        """

    base_3 = """ 
        Final Answer:
        """

    # Abalation prompts

    option_1_vs_system = """
        The given data is noisy and may contain artifacts. You should behave like a Bayesian observer and take into account prior and likelihood in your predictions.     
    """

    option_1_ns_system = """
        The given data is noisy and may contain artifacts. For 10 previous observations, the values were observed to lie in the range of {value_range}.     
    """

    option_1_ps_system = """
        The given data is noisy and may contain artifacts. For 10 previous observations, the values were observed to lie in the range of {value_range}.     
    """

    def __init__(
        self,
        experiment_mode="text_image",
        image_mode="normal",
        system_prompt="You are a lines length ratio estimator.",
        verbal_steer=False,
        numerical_steer=False,
        prior_steer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experiment_mode = experiment_mode
        self.multi_modal = self.experiment_mode != "text"
        self.image_mode = image_mode
        self.verbal_steer = verbal_steer
        self.numerical_steer = numerical_steer
        self.prior_steer = prior_steer
        self.system_prompt = system_prompt

        if self.experiment_mode == "text_image":
            self.system_prompt += self.base_1_text_image_system
            self.user_prompt_zero_shot = self.base_1_text_image
        if self.experiment_mode == "text":
            self.system_prompt += self.base_1_text_system
            self.user_prompt_zero_shot = self.base_1_text
        if self.experiment_mode == "image":
            self.system_prompt += self.base_1_image_system
            self.user_prompt_zero_shot = self.base_1_image

        if self.verbal_steer:
            self.system_prompt += self.option_1_vs_system
        if self.numerical_steer:
            self.system_prompt += self.option_1_ns_system
        if self.prior_steer:
            self.system_prompt += self.option_1_ps_system

        self.system_prompt += self.base_3_system
        self.user_prompt_zero_shot += self.base_3
        self.system_prompt = textwrap.dedent(self.system_prompt)
        self.user_prompt_zero_shot = textwrap.dedent(self.user_prompt_zero_shot)

        self.experiment_module = (
            f"experiments.{self.experiment_type}.{self.experiment_name}.experiment_run"
        )
        self.experiment_path = (
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/runs/{self.exp_id}/"
        )
        self.experiment_files = [
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/0.1_0.5/line_len_ratio_0.1_0.5_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/0.3_0.8/line_len_ratio_0.3_0.8_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/0.5_0.9/line_len_ratio_0.5_0.9_100samples.csv",
        ]


class ImageMazeDistance(CommonConfig):
    experiment_type = "text_image"
    experiment_name = "maze_distance"

    base_1_text_system = """
        You are given a text-based description of a path within a square boundary. 

        Estimate the straight-line (Euclidean) distance in terms of units between the start and the end of the path.
        """

    base_1_image_system = """
        You are given an image of a path within a square boundary. 

        In the image the start is marked with a green dot and the end is marked with a red cross. The background grid in the image consists of unit squares, meaning each grid cell corresponds to one unit in length along both axes. 
        
        Estimate the straight-line (Euclidean) distance in terms of units between the start and the end of the path.
        """

    base_1_text_image_system = """
        You are given both an image and a text-based description of a path within a square boundary.
        
        In the image the start is marked with a green dot and the end is marked with a red cross. The background grid in the image consists of unit squares, meaning each grid cell corresponds to one unit in length along both axes. 
        
        Estimate the straight-line (Euclidean) distance in terms of units between the start and the end of the path.
        """

    base_3_system = """ 
        Do not explain or reason. Only output the final answer as a number after Final Answer:
        """

    base_1_text = """
        The text description of the path is:
        {text_representation}
        """

    base_1_image = """
        Consider the image provided.
        """

    base_1_text_image = """ 
        Consider the image provided and the text description of the path.

        {text_representation}
        """

    base_3 = """ 
        Final Answer:
        """

    # Ablation prompts
    option_1_vs_system = """
        The given data is noisy and may contain artifacts. You should behave like a Bayesian observer and take into account prior and likelihood in your predictions.     
    """

    option_1_ns_system = """
        The given data is noisy and may contain artifacts. For 10 previous observations, the values were observed to lie in the range of {value_range}.     
    """

    option_1_ps_system = """
        The given data is noisy and may contain artifacts. For 10 previous observations, the values were observed to lie in the range of {value_range}.     
    """

    def __init__(
        self,
        experiment_mode="text_image",
        text_mode="path_text",
        image_mode="normal",
        provide_answer=False,
        system_prompt="You are a maze distance estimator.",
        verbal_steer=False,
        numerical_steer=False,
        prior_steer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.experiment_mode = experiment_mode
        self.multi_modal = self.experiment_mode != "text"
        self.provide_answer = provide_answer
        self.text_mode = text_mode
        self.image_mode = image_mode
        self.verbal_steer = verbal_steer
        self.numerical_steer = numerical_steer
        self.prior_steer = prior_steer
        self.system_prompt = system_prompt

        if self.experiment_mode == "text_image":
            self.system_prompt += self.base_1_text_image_system
            self.user_prompt_zero_shot = self.base_1_text_image
        if self.experiment_mode == "text":
            self.system_prompt += self.base_1_text_system
            self.user_prompt_zero_shot = self.base_1_text
        if self.experiment_mode == "image":
            self.system_prompt += self.base_1_image_system
            self.user_prompt_zero_shot = self.base_1_image

        if self.verbal_steer:
            self.system_prompt += self.option_1_vs_system
        if self.numerical_steer:
            self.system_prompt += self.option_1_ns_system
        if self.prior_steer:
            self.system_prompt += self.option_1_ps_system

        self.system_prompt += self.base_3_system
        self.user_prompt_zero_shot += self.base_3
        self.system_prompt = textwrap.dedent(self.system_prompt)
        self.user_prompt_zero_shot = textwrap.dedent(self.user_prompt_zero_shot)

        self.experiment_module = (
            f"experiments.{self.experiment_type}.{self.experiment_name}.experiment_run"
        )
        self.experiment_path = (
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/runs/{self.exp_id}/"
        )
        self.experiment_files = [
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/1.0_5.0/maze_distance_1.0_5.0_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/3.0_7.0/maze_distance_3.0_7.0_100samples.csv",
            EXP_FILE_DIR
            / f"{self.experiment_type}/{self.experiment_name}/data/experiment_files/5.0_9.0/maze_distance_5.0_9.0_100samples.csv",
        ]
