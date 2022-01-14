using System.Threading;
using System.Threading.Tasks;
using Elsa.Services;
using Elsa.WorkflowSettings.Persistence.EntityFramework.Core.Services;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace Elsa.WorkflowSettings.Persistence.EntityFramework.Core.StartupTasks
{
    /// <summary>
    /// Executes EF Core migrations.
    /// </summary>
    public class RunMigrations : IStartupTask
    {
        protected readonly IServiceScopeFactory _scopeFactory;

        public RunMigrations(IServiceScopeFactory scopeFactory)
        {
            _scopeFactory = scopeFactory;
        }

        public int Order => 0;

        public virtual async Task ExecuteAsync(CancellationToken cancellationToken = default)
        {
           await ExecuteInternalAsync(cancellationToken);
        }

        protected async Task ExecuteInternalAsync(CancellationToken cancellationToken = default, IServiceScope? serviceScope = default)
        {
            using var scope = serviceScope ?? _scopeFactory.CreateScope();

            var dbContextFactory = scope.ServiceProvider.GetRequiredService<IWorkflowSettingsContextFactory>();

            await using var dbContext = dbContextFactory.CreateDbContext();
            await dbContext.Database.MigrateAsync(cancellationToken);
            await dbContext.DisposeAsync();
        }
    }
}