from src.adaptive_rag.router import (
    QueryRouter,
    ROUTE_DIRECT,
    ROUTE_RETRIEVE_STANDARD,
    ROUTE_RETRIEVE_MULTI_HOP,
    ROUTE_REJECT
)

def test_router_rejection():
    router = QueryRouter()
    res = router.route("How to configure Hibernate entities in FastAPI?")
    assert res["route"] == ROUTE_REJECT
    assert "hibernate" in res["reason"].lower()

def test_router_multihop():
    router = QueryRouter()
    res = router.route("How do yield dependencies interact with background tasks?")
    assert res["route"] == ROUTE_RETRIEVE_MULTI_HOP

def test_router_standard():
    router = QueryRouter()
    res = router.route("What is status_code parameter in FastAPI?")
    assert res["route"] == ROUTE_RETRIEVE_STANDARD

def test_router_direct():
    router = QueryRouter()
    res = router.route("hello")
    assert res["route"] == ROUTE_DIRECT
