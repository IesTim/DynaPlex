#include <gtest/gtest.h>
#include "testutils.h"

namespace DynaPlex::Tests {

    TEST(dual_sourcing_backlog, mdp_config_0)
    {
        Tester tester{};
        tester.ExecuteTest("dual_sourcing_backlog", "mdp_config_0.json");
    }

}