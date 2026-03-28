from public_api.core.pipelines.base import BasePipelineExecutor


def test_make_chunks_basic_overlap():
    text = "x" * 100
    base = BasePipelineExecutor
    chunks = base.make_chunks(text, chunk_size=30, overlap=10)

    assert chunks[0].start == 0 and chunks[0].end == 30
    assert chunks[1].start == 20 and chunks[1].end == 50
    assert chunks[-1].end == 100
    # coverage: no gaps
    assert chunks[0].start == 0
    assert all(chunks[i].end >= chunks[i + 1].start for i in range(len(chunks) - 1))
