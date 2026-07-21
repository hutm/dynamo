# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo import gms_router_policy
from dynamo.gms_router_policy import (

    maybe_fetch_gms_placement,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_router_policy_exposes_no_gms_control_endpoint():
    assert not hasattr(gms_router_policy, "GMS_CONTROL_ENDPOINT")
    assert not hasattr(gms_router_policy, "make_gms_control_handler")


@pytest.mark.asyncio
async def test_missing_local_gms_socket_leaves_request_on_vanilla_path():
    request = {
        "prompt": "hello",
        "gms_placement": {
            "source_nixl_agent_name": "source",
            "source_nixl_agent_metadata_hex": "00",
            "hashes": ["00"],
            "descriptors": [None],
        },
    }

    result = await maybe_fetch_gms_placement(request, None)

    assert result is None
    assert request["prompt"] == "hello"
    assert "gms_placement" in request


def test_router_gms_decode_transfer_arg_defaults_off():
    from dynamo.router.args import parse_args

    config = parse_args(["--endpoint", "ns.comp.generate"])

    assert config.router_gms_decode_transfer is False
    assert config.kv_router_kwargs()["router_gms_decode_transfer"] is False


def test_router_gms_decode_transfer_arg_can_opt_in():
    from dynamo.router.args import parse_args

    config = parse_args(
        ["--endpoint", "ns.comp.generate", "--router-gms-decode-transfer"]
    )

    assert config.router_gms_decode_transfer is True
    assert config.kv_router_kwargs()["router_gms_decode_transfer"] is True


# NOTE: GMS placement-router publishing was dropped when rebasing onto upstream
# main (its indexer refactor is incompatible with our gms_placement threading).
# DynamoGmsPlacementPublisher is retained as an inert no-op, so the descriptor-
# normalization / publish tests that asserted event emission were removed.
