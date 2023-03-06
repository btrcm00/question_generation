# Question Generation

## Introduce

- Question generation is the task of generating questions from a passage with marked answer, clue and question type that you want to generate.

- The goal of this task is to generate more data for QA tasks.

- Model: [BERT2BERT](https://huggingface.co/blog/warm-starting-encoder-decoder) and BERT2BERT+[Pointer](https://arxiv.org/abs/1704.04368)
- Dataset:
  - Train: 55000
  - Dev: 2000
  - Test: 2000

## How to use

- load pretrained tokenizer and model from checkpoint_folder
- specify training_config when loading pretrained model, `use_pointer` determine whether checkpoint use Pointer or not, `logging_dir=""`

```
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_folder)
    model = QG_EncoderDecoderModel.from_pretrained(checkpoint_folder, training_config={"use_pointer": True, "logging_dir": ""})
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
        ex: <PERSON>Nguyễn Trần Hiếu</PERSON>, <ORG>TMT</ORG>, ...
    - Choose question type that you want [What, Who, When, ...]

- You can see code to predict in /inference/predict.py

## API (updating ...)
