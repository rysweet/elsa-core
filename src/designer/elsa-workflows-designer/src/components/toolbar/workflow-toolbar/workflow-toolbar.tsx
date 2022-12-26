import {Component, State, h} from '@stencil/core';
import Container from 'typedi';
import {NotificationEventTypes} from '../../../modules/notifications/event-types';
import {EventBus} from '../../../services';
import toolbarComponentStore from "../../../data/toolbar-component-store";
import notificationService from '../../../modules/notifications/notification-service';
import notificationStore from "../../../modules/notifications/notification-store";

@Component({
  tag: 'elsa-workflow-toolbar',
})
export class WorkflowToolbar {
  @State() public modalState: boolean = false;
  private readonly eventBus: EventBus;
  static NotificationService = notificationService;
  constructor() {
    this.eventBus = Container.get(EventBus);
  }

  onNotificationClick = e => {
    e.stopPropagation();
    // this.eventBus.emit(NotificationEventTypes.Toggle, this);
    WorkflowToolbar.NotificationService.toogleNotification();
    this.modalState = !this.modalState;
  };

  render() {
    const { notifications, infoPanelBoolean } = notificationStore;
    return (
      <div>
      <nav class="bg-gray-800">
        <div class="mx-auto px-2 sm:px-6 lg:px-6">
          <div class="relative flex items-center justify-end h-16">
            <div class="inset-y-0 right-0 flex items-center pr-2 sm:static sm:inset-auto sm:ml-6 sm:pr-0 z-40">
              {/* Notifications*/}
              <button
                onClick={e => this.onNotificationClick(e)}
                type="button"
                class="bg-gray-800 p-1 rounded-full text-gray-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-white mr-4"
              >
                <span class="sr-only">View notifications</span>
                <svg class="h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                  <path
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    stroke-width="2"
                    d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9"
                  />
                </svg>
              </button>

              {toolbarComponentStore.components.map(component => (
                <div class="flex-shrink-0 mr-4">
                  {component()}
                </div>
              ))}

              {/* Menu */}
              <elsa-workflow-toolbar-menu/>
            </div>
          </div>
        </div>
      </nav>
        <elsa-notifications-manager modalState={this.modalState}></elsa-notifications-manager>
        {notifications && notifications.map(item => {
          <elsa-awhile-notifications notification={item}></elsa-awhile-notifications>
        })}
        </div>
    );
  }
}
