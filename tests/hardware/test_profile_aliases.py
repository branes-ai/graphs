"""Tests for profile-as-SKU registry aliases (issue #136 Phase 1).

Locks in the contract:
- ``list_all_mappers()`` returns silicon-bin names only (backward-compat)
- ``list_all_skus()`` adds profile aliases of the form ``"{silicon}@{profile}"``
- ``get_mapper_by_name`` resolves both ``@`` and dashed alias forms
- Bare silicon-bin name still uses the default thermal profile
- ``thermal_profile`` kwarg always wins over an alias-derived profile
- Invalid profile names return ``None``, not raise
"""

from graphs.hardware.mappers import (
    get_mapper_by_name,
    list_all_mappers,
    list_all_skus,
)
from graphs.hardware.mappers import _resolve_profile_alias


class TestBackwardCompat:
    def test_list_all_mappers_returns_silicon_bins_only(self):
        names = list_all_mappers()
        assert len(names) > 0
        # No alias forms in this list -- preserves legacy callers.
        assert all("@" not in n for n in names)

    def test_bare_silicon_name_uses_default_profile(self):
        # get_mapper_by_name without a kwarg should return the default
        # thermal profile, identical to pre-#136 behavior.
        m = get_mapper_by_name("Jetson-Orin-Nano-8GB")
        assert m is not None
        # 15W is the default profile on Orin Nano (set in the resource model).
        assert m.thermal_profile == "15W"

    def test_legacy_thermal_profile_kwarg_still_works(self):
        # Pre-#136 callers passing `thermal_profile=` to a bare name should
        # behave unchanged.
        m = get_mapper_by_name("Jetson-Orin-Nano-8GB", thermal_profile="7W")
        assert m is not None
        assert m.thermal_profile == "7W"


class TestAliasResolution:
    """Both alias forms (@ and dashed) resolve correctly."""

    def test_at_form_resolves(self):
        m = get_mapper_by_name("Jetson-Orin-Nano-8GB@7W")
        assert m is not None
        assert m.thermal_profile == "7W"
        # Operational fields reflect the chosen profile.
        tdp = m.resource_model.thermal_operating_points[m.thermal_profile].tdp_watts
        assert tdp == 7.0

    def test_dashed_form_resolves(self):
        m = get_mapper_by_name("Jetson-Orin-Nano-8GB-15W")
        assert m is not None
        assert m.thermal_profile == "15W"
        tdp = m.resource_model.thermal_operating_points[m.thermal_profile].tdp_watts
        assert tdp == 15.0

    def test_at_and_dashed_forms_return_equivalent_mappers(self):
        m_at = get_mapper_by_name("Jetson-Orin-Nano-8GB@MAXN")
        m_dashed = get_mapper_by_name("Jetson-Orin-Nano-8GB-MAXN")
        assert m_at is not None
        assert m_dashed is not None
        assert m_at.thermal_profile == m_dashed.thermal_profile == "MAXN"

    def test_alias_resolution_works_for_orin_agx(self):
        # Orin AGX 64GB has 4 thermal profiles (15W, 30W, 50W, MAXN per
        # nvpmodel). Spot-check that the alias machinery works there too.
        m = get_mapper_by_name("Jetson-Orin-AGX-64GB@30W")
        assert m is not None
        assert m.thermal_profile == "30W"


class TestInvalidAliases:
    def test_unknown_silicon_returns_none(self):
        assert get_mapper_by_name("Definitely-Not-A-Chip") is None
        assert get_mapper_by_name("Definitely-Not-A-Chip@7W") is None

    def test_unknown_profile_returns_none(self):
        # Silicon exists, but the profile suffix is bogus.
        assert get_mapper_by_name("Jetson-Orin-Nano-8GB@99W-imaginary") is None
        assert get_mapper_by_name("Jetson-Orin-Nano-8GB-99W-imaginary") is None

    def test_at_form_does_not_fall_through_to_dashed(self):
        # If the user types the @ form with an invalid profile, we must NOT
        # silently fall through and try to interpret the whole string as a
        # dashed alias. The @ is an explicit user signal of intent.
        assert (
            get_mapper_by_name("Jetson-Orin-Nano-8GB@imaginary")
            is None
        )


class TestKwargPrecedence:
    def test_kwarg_overrides_alias_profile(self):
        # User typed @7W in the name AND passed thermal_profile=MAXN.
        # The kwarg is the more explicit signal -- it wins.
        m = get_mapper_by_name(
            "Jetson-Orin-Nano-8GB@7W", thermal_profile="MAXN"
        )
        assert m is not None
        assert m.thermal_profile == "MAXN"

    def test_kwarg_overrides_dashed_alias_profile(self):
        m = get_mapper_by_name(
            "Jetson-Orin-Nano-8GB-7W", thermal_profile="MAXN"
        )
        assert m is not None
        assert m.thermal_profile == "MAXN"

    def test_empty_string_kwarg_does_not_silently_fall_back_to_alias(self):
        # ``thermal_profile=""`` is an explicit (if invalid) user signal,
        # not a "not provided" sentinel. Precedence is checked with
        # ``is not None`` so the empty string takes priority over the
        # alias-encoded profile. The empty string isn't a registered
        # profile, so the underlying factory call fails the lookup and
        # the resource model raises -- but we should NOT have silently
        # routed the request to the alias-encoded "7W".
        try:
            m = get_mapper_by_name(
                "Jetson-Orin-Nano-8GB@7W", thermal_profile=""
            )
        except (ValueError, KeyError):
            # Expected: empty string isn't a valid profile.
            return
        # If the factory was tolerant of empty-string and didn't raise,
        # at minimum the alias-encoded "7W" should NOT have leaked through.
        assert m is None or m.thermal_profile != "7W"


class TestSkuEnumeration:
    def test_list_all_skus_includes_silicon_and_aliases(self):
        skus = list_all_skus()
        silicon = list_all_mappers()
        # Every silicon-bin name appears.
        for s in silicon:
            assert s in skus
        # And we have strictly more entries (every silicon-bin has at least
        # one thermal profile).
        assert len(skus) > len(silicon)

    def test_list_all_skus_alias_format_uses_at_separator(self):
        skus = list_all_skus()
        # Every alias entry should contain exactly one '@'. (Silicon-bin
        # names never contain '@'.)
        aliases = [s for s in skus if "@" in s]
        assert len(aliases) > 0
        for a in aliases:
            assert a.count("@") == 1

    def test_list_all_skus_orin_nano_has_four_profiles(self):
        # Orin Nano Super resource model registers four canonical modes:
        # 7W (battery), 15W (Phase B baseline), 25W (production cap with
        # DVFS), MAXN (developer/burst, DVFS off). Anchored on the user
        # directive captured in #136. Should this expand in the future
        # (e.g., a 50W mode for some hypothetical refresh), update this
        # test alongside the resource model -- the assertion is exact by
        # design to catch silent profile additions/removals.
        nano_aliases = [
            s for s in list_all_skus() if s.startswith("Jetson-Orin-Nano-8GB@")
        ]
        profiles = sorted(s.split("@", 1)[1] for s in nano_aliases)
        assert profiles == ["15W", "25W", "7W", "MAXN"]

    def test_orin_nano_25w_and_maxn_share_tdp_but_differ_in_dvfs(self):
        # 25W mode and MAXN mode both cap at 25W TDP on the Super silicon
        # but have different DVFS semantics: 25W allows scaling down for
        # power-aware workloads, MAXN locks at boost. Spec invariant:
        # both modes have tdp_watts=25.0 but different cooling and DVFS
        # configurations.
        m_25w = get_mapper_by_name("Jetson-Orin-Nano-8GB@25W")
        m_maxn = get_mapper_by_name("Jetson-Orin-Nano-8GB@MAXN")
        assert m_25w is not None and m_maxn is not None
        rm_25w = m_25w.resource_model.thermal_operating_points["25W"]
        rm_maxn = m_maxn.resource_model.thermal_operating_points["MAXN"]
        assert rm_25w.tdp_watts == rm_maxn.tdp_watts == 25.0
        # Distinct entries -- not aliases of the same object.
        assert rm_25w.name != rm_maxn.name

    def test_include_profile_aliases_false_returns_silicon_only(self):
        skus = list_all_skus(include_profile_aliases=False)
        assert all("@" not in s for s in skus)
        assert sorted(skus) == sorted(list_all_mappers())

    def test_every_alias_resolves(self):
        # Sanity: every entry in list_all_skus() can be looked up by
        # get_mapper_by_name. Guards against alias-format drift.
        for sku in list_all_skus():
            m = get_mapper_by_name(sku)
            assert m is not None, f"alias did not resolve: {sku}"


class TestResolveProfileAliasInternal:
    """Direct tests for the private _resolve_profile_alias function. Useful
    because the public get_mapper_by_name swallows the silicon/profile pair
    and only exposes the resulting mapper instance."""

    def test_at_form_returns_silicon_and_profile(self):
        result = _resolve_profile_alias("Jetson-Orin-Nano-8GB@7W")
        assert result == ("Jetson-Orin-Nano-8GB", "7W")

    def test_dashed_form_returns_silicon_and_profile(self):
        result = _resolve_profile_alias("Jetson-Orin-Nano-8GB-15W")
        assert result == ("Jetson-Orin-Nano-8GB", "15W")

    def test_unknown_returns_none(self):
        assert _resolve_profile_alias("not-a-thing") is None
        assert _resolve_profile_alias("Jetson-Orin-Nano-8GB@bogus") is None
