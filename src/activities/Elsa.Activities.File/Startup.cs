using Elsa.Attributes;
using Elsa.Options;
using Elsa.Services.Startup;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace Elsa.Activities.File
{
    [Feature("File")]
    public class Startup : StartupBase
    {
        public override void ConfigureElsa(ElsaOptionsBuilder elsa, IConfiguration configuration)
        {
            var multitenancyEnabled = configuration.GetValue<bool>("Elsa:MultiTenancy");

            elsa.AddFileActivities(options => options.MultitenancyEnabled = multitenancyEnabled);
        }
    }
}