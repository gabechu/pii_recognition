def test_sidebar(client):
    actual = client.get("/sidebar")
    assert actual.status_code == 200
    assert actual.data is not None
