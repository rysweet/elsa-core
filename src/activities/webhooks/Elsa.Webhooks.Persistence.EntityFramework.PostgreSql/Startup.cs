using System;
using Elsa.Abstractions.MultiTenancy;
using Elsa.Attributes;
using Elsa.Webhooks.Persistence.EntityFramework.Core;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace Elsa.Webhooks.Persistence.EntityFramework.PostgreSql
{
    [Feature("Webhooks:EntityFrameworkCore:PostgreSql")]
    public class Startup : EntityFrameworkWebhookStartupBase
    {
        protected override string ProviderName => "PostgreSql";
        protected override void Configure(DbContextOptionsBuilder options, string connectionString) => options.UseWebhookPostgreSql(connectionString);
        protected override void ConfigureForMultitenancy(DbContextOptionsBuilder options, IServiceProvider serviceProvider)
        {
            var tenantProvider = serviceProvider.GetRequiredService<ITenantProvider>();

            var connectionString = tenantProvider.GetCurrentTenant().ConnectionString;

            options.UseWebhookPostgreSql(connectionString);
        }
    }
}