import pytest

from src.utils.data_utils import ArticleLoader


# 定义 fixture 来初始化 ArticleLoader 实例
@pytest.fixture
def article_loader():
    return ArticleLoader("./data/articles.json")


def test_all_articles_loaded(article_loader):
    assert len(list(article_loader.all_articles.keys())) == 250


def test_load_article(article_loader):
    assert (
        article_loader.load_article("114")
        == "《中华人民共和国刑法》第一百一十四条：放火、决水、爆炸以及投放毒害性、放射性、传染病病原体等物质或者以其他危险方法危害公共安全，尚未造成严重后果的，处三年以上十年以下有期徒刑。"
    )
    with pytest.raises(KeyError):
        article_loader.load_article(114)


def test_get_article_id(article_loader):
    assert (
        article_loader.get_article_id(
            "《中华人民共和国刑法》第一百一十四条：放火、决水、爆炸以及投放毒害性、放射性、传染病病原体等物质或者以其他危险方法危害公共安全，尚未造成严重后果的，处三年以上十年以下有期徒刑。"
        )
        == "114"
    )
    with pytest.raises(ValueError):
        article_loader.get_article_id("Non-existent article")
