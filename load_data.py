# -*- coding: utf-8 -*-
# @Time    : 4/10/2024 11:40 AM
# @Author  : yuzhn
# @File    : load_data.py
# @Software: PyCharm
import argparse

import pandas as pd
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'The information about data, models and methods')
    parser.add_argument("--model_name", type = str,
                        default =
                        ["gpt-3.5-turbo-0613", "gpt-4-0613",
                         "text-davinci-003", "text-curie-001",
                         "text-babbage-001"][1])
    parser.add_argument("--data_path",
                        type = str,
                        default = ["data/data_en_zh.dict", ][0])
    parser.add_argument("--encode_method",
                        type = str,
                        default = ["unchange", "ascii", "caesar",
                                   "baseline", "unicode",
                                   "morse", "atbash", "utf",
                                   "gbk"][
                            0])  # unchange is the SelfCipher, baseline is the vanilla

    parser.add_argument("--instruction_type", type = str,
                        default = ["Crimes_And_Illegal_Activities",
                                   "Ethics_And_Morality",
                                   "Inquiry_With_Unsafe_Opinion", "Insult",
                                   "Mental_Health", "Physical_Harm",
                                   "Privacy_And_Property", "Reverse_Exposure",
                                   "Role_Play_Instruction",
                                   "Unfairness_And_Discrimination",
                                   "Unsafe_Instruction_Topic"][0])
    parser.add_argument("--use_system_role", type = bool, default = True)
    parser.add_argument("--use_demonstrations", type = bool, default = True)
    parser.add_argument("--demonstration_toxicity",
                        type = str,
                        default = ["toxic", "harmless"][
                            0])  # harmless means that use the safe demonstrations
    parser.add_argument("--language", type = str, default = ["zh", "en"][-1])

    parser.add_argument("--debug", type = bool, default = False)
    parser.add_argument("--debug_num", type = int, default = 3)
    parser.add_argument("--temperature", type = float, default = 0)
    args = parser.parse_args()
    model_name = args.model_name.split('/')[-1]
    attribution = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(model_name,
                                                         args.data_path.split("/")[0],
                                                         args.instruction_type.lower().replace("_", "-"),
                                                         args.encode_method,
                                                         args.demonstration_toxicity,
                                                         args.language,
                                                         args.use_system_role,
                                                         args.use_demonstrations,
                                                         args.debug_num,
                                                         args.temperature, )
    saved_path = "saved_results/{}_results.list".format(attribution)  # the path to save the conversations
    data = torch.load(saved_path)[1:]
    data_list = [i["response"] for i in data]
    # save the list to dataframe and then to csv

    df = pd.DataFrame(data_list, columns = ["text"])
    df.to_csv(f"saved_results/cipher_{model_name}_results.csv", index = False)
