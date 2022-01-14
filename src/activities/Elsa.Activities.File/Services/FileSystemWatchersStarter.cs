using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using AutoMapper;
using Elsa.Abstractions.MultiTenancy;
using Elsa.Activities.File.Bookmarks;
using Elsa.Activities.File.Models;
using Elsa.Models;
using Elsa.Services;
using Elsa.Services.Models;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Elsa.Activities.File.Services
{
    public class FileSystemWatchersStarter : IFileSystemWatchersStarter
    {
        protected readonly SemaphoreSlim _semaphore = new(1);
        protected readonly ICollection<Tuple<FileSystemWatcher, Tenant?>> _watchers;
        protected readonly IServiceScopeFactory _scopeFactory;
        private readonly ILogger<FileSystemWatchersStarter> _logger;
        private readonly IMapper _mapper;
        private readonly Scoped<IWorkflowLaunchpad> _workflowLaunchpad;

        public FileSystemWatchersStarter(
            ILogger<FileSystemWatchersStarter> logger,
            IMapper mapper,
            IServiceScopeFactory scopeFactory,
            Scoped<IWorkflowLaunchpad> workflowLaunchpad)
        {
            _logger = logger;
            _mapper = mapper;
            _scopeFactory = scopeFactory;
            _watchers = new List<Tuple<FileSystemWatcher, Tenant?>>();
            _workflowLaunchpad = workflowLaunchpad;
        }

        public virtual async Task CreateAndAddWatchersAsync(CancellationToken cancellationToken = default)
        {
            await _semaphore.WaitAsync(cancellationToken);

            try
            {
                if (_watchers.Any())
                {
                    foreach (var (watcher, _) in _watchers)
                        watcher.Dispose();

                    _watchers.Clear();
                }

                using var scope = _scopeFactory.CreateScope();

                var activities = GetActivityInstancesAsync(scope.ServiceProvider, cancellationToken);
                await foreach (var a in activities.WithCancellation(cancellationToken))
                {
                    var changeTypes = await a.EvaluatePropertyValueAsync(x => x.ChangeTypes, cancellationToken);
                    var notifyFilters = await a.EvaluatePropertyValueAsync(x => x.NotifyFilters, cancellationToken);
                    var path = await a.EvaluatePropertyValueAsync(x => x.Path, cancellationToken);
                    var pattern = await a.EvaluatePropertyValueAsync(x => x.Pattern, cancellationToken);
                    CreateAndAddWatcher(path, pattern, changeTypes, notifyFilters);
                }
            }
            finally
            {
                _semaphore.Release();
            }
        }

        protected void CreateAndAddWatcher(string? path, string? pattern, WatcherChangeTypes changeTypes, NotifyFilters notifyFilters, Tenant? tenant = default)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("File watcher path must not be null or empty");

            EnsurePathExists(path);

            var watcher = new FileSystemWatcher()
            {
                Path = path,
                Filter = pattern,
                NotifyFilter = notifyFilters
            };

            if (changeTypes == WatcherChangeTypes.Created || changeTypes == WatcherChangeTypes.All)
                watcher.Created += FileCreated;

            if (changeTypes == WatcherChangeTypes.Changed || changeTypes == WatcherChangeTypes.All)
                watcher.Changed += FileChanged;

            if (changeTypes == WatcherChangeTypes.Deleted || changeTypes == WatcherChangeTypes.All)
                watcher.Deleted += FileDeleted;

            if (changeTypes == WatcherChangeTypes.Renamed || changeTypes == WatcherChangeTypes.All)
                watcher.Renamed += FileRenamed;

            watcher.EnableRaisingEvents = true;
            _watchers.Add(Tuple.Create(watcher, tenant));
        }

        protected async IAsyncEnumerable<IActivityBlueprintWrapper<WatchDirectory>> GetActivityInstancesAsync(IServiceProvider serviceProvider, [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var workflowRegistry = serviceProvider.GetRequiredService<IWorkflowRegistry>();
            var workflowBlueprintReflector = serviceProvider.GetRequiredService<IWorkflowBlueprintReflector>();
            var workflows = await workflowRegistry.ListActiveAsync(cancellationToken);

            var query = from workflow in workflows
                        from activity in workflow.Activities
                        where activity.Type == nameof(WatchDirectory)
                        select workflow;

            foreach (var workflow in query)
            {
                var workflowBlueprintWrapper = await workflowBlueprintReflector.ReflectAsync(serviceProvider, workflow, cancellationToken);

                foreach (var activity in workflowBlueprintWrapper.Filter<WatchDirectory>())
                {
                    yield return activity;
                }
            }
        }

        private void EnsurePathExists(string path)
        {
            _logger.LogDebug("Checking ${Path} exists", path);

            if (Directory.Exists(path))
                return;

            _logger.LogInformation("Creating directory {Path}", path);
            Directory.CreateDirectory(path);
        }

        #region Watcher delegates
        private void FileCreated(object sender, FileSystemEventArgs e)
        {
            StartWorkflow((FileSystemWatcher)sender, e);
        }

        private void FileChanged(object sender, FileSystemEventArgs e)
        {
            if (e.ChangeType != WatcherChangeTypes.Changed)
            {
                return;
            }

            StartWorkflow((FileSystemWatcher)sender, e);
        }

        private void FileDeleted(object sender, FileSystemEventArgs e)
        {
            StartWorkflow((FileSystemWatcher)sender, e);
        }

        private void FileRenamed(object sender, RenamedEventArgs e)
        {
            StartWorkflow((FileSystemWatcher)sender, e);
        }

        private async void StartWorkflow(FileSystemWatcher watcher, FileSystemEventArgs e)
        {
            var changeTypes = e.ChangeType;
            var notifyFilter = watcher.NotifyFilter;
            var path = watcher.Path;
            var pattern = watcher.Filter;

            var model = _mapper.Map<FileSystemEvent>(e);
            var bookmark = new FileSystemEventBookmark(path, pattern, changeTypes, notifyFilter);
            var launchContext = new WorkflowsQuery(nameof(WatchDirectory), bookmark);

            var tenant = _watchers.FirstOrDefault(x => x.Item1 == watcher)?.Item2;

            if (tenant != null)
                await _workflowLaunchpad.UseServiceWithTenantAsync(s => s.CollectAndDispatchWorkflowsAsync(launchContext, new WorkflowInput(model)), tenant);
            else
                await _workflowLaunchpad.UseServiceAsync(s => s.CollectAndDispatchWorkflowsAsync(launchContext, new WorkflowInput(model)));
        }

        #endregion
    }
}