# Question Generation

A Pipeline from data construction to model deployment for Question-generation task using BARTPho.
UPDATING: Add ONNX converter to pipeline for increasing inference performance.

## Introduce

- Question generation is the task of generating questions from a passage with marked answer, clue and question type that you want to generate.

- The goal of this task is to generate more data for QA tasks.

- Model: BartPho+[Pointer](https://arxiv.org/abs/1704.04368)
- Dataset:
  - Train: ~340000
  - Dev: ~12000
  - Test: ~12000

## How to use

- load pretrained tokenizer and model from checkpoint_folder
- specify training_config when loading pretrained model, `use_pointer` determine whether checkpoint use Pointer or not, `logging_dir=""`

```
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    model = BartPhoPointer.from_pretrained(checkpoint_folder, training_config={"use_pointer": True, "logging_dir": ""})
```

- example form:

```
    {
        'passage_ans_clue': '<CLUE> Công Phượng đã bị trượt khi thi vào lò đào tạo bóng đá trẻ của câu lạc bộ bóng đá <ORG> Sông Lam Nghệ An </ORG> bởi thể hình quá nhỏ con </CLUE> . Tưởng chừng như hy vọng trở thành cầu thủ chuyên nghiệp của Công Phượng đã chấm dứt thì cơ hội khác lại mở ra khi Phượng tình cờ nghe được thông tin tuyển sinh của học viện HAGL - Arsenal - JMG trên tivi. Vậy là gia đình đã phải gác lại ý định sửa nhà để dành dụm, thậm chí đã phải vay thêm tiền đưa con lên Gia Lai thử sức. Trải qua 1 vòng tuyển chọn gắt gao, Nguyễn Công Phượng là một trong các cầu thủ nhí được ghi danh vào Học viện bóng đá HAGL-Arsenal JMG khóa 1.',
        'ques_type': 'What',
    }
```

    - Mark clue in passage with <CLUE> and </CLUE>
    - Mark answer in passage with tags as <ORG></ORG>, <PERSON></PERSON>, <LOC></LOC>, ... corresponding to NER tag of answer
        ex: <ORG>TMT</ORG>, ...
    - Choose question type that you want [What, Who, When, ...]

- You can see code to predict in /inference/predict.py

## API (updating ...)

## How to run ?

- Clone repo
- Create .env file and add some environment variables
    - if you want to pull data or checkpoint, you have to define STORAGE in .env file

Basic command: `bash pipeline/pipeline.sh <mode> <args>`
- `<mode>` is one of [train, sampling, prepare_data, all]
- `<args>` is the args to pass to module
- Read pipeline/pipeline.sh file for understanding detail flow

!!! .env file in source directory contain all of pipeline environment variables
And pipeline/scripts/env/*.env files contain specific variables for each sub module

examle:
`bash pipeline/pipeline.sh train`
`bash pipeline/pipeline.sh api --parallel_input_processing`
