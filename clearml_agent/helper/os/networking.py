import psutil


class TcpPorts(object):
    default_remap_port_min = 10000
    default_remap_port_max = 60000

    def __init__(self, allowed_ports_range=None):
        self._used_ports = sorted([i.laddr.port for i in psutil.net_connections()])
        self.allowed_ports_range = list(allowed_ports_range or [])

    def check_tcp_port_available(self, port: int, remember_port: bool = True) -> bool:
        """
        return True if the port is available
        :param port: port number
        :param remember_port: if True add the port into the used ports list
        :return: True port is available
        """
        if self.allowed_ports_range and (port < min(self.allowed_ports_range) or port > max(self.allowed_ports_range)):
            return False
        if port in self._used_ports:
            return False
        if remember_port:
            self._used_ports.append(port)
        return True

    def find_port_range(self, number_of_ports: int, remember_port: bool = True,
                        range_min: int = None, range_max: int = None) -> list:
        """
        Find an available ports range for the number of ports requested. min and max are defined as
         TcpPorts.remap_port_min as TcpPorts.remap_port_max, and will be affected by the allowed port
         range provided on init.
        """
        if range_min is None:
            range_min = min(self.allowed_ports_range) if self.allowed_ports_range else self.default_remap_port_min
        if range_max is None:
            range_max = max(self.allowed_ports_range) if self.allowed_ports_range else self.default_remap_port_max

        ports = (i for i in range(range_min, range_max) if i not in self._used_ports)
        new_allocation = []
        for p in ports:
            # find consecutive ports
            if new_allocation and (new_allocation[-1]+1) != p:
                new_allocation = []

            new_allocation.append(p)
            if len(new_allocation) == number_of_ports:
                break

        # check if we found enough
        if len(new_allocation) != number_of_ports:
            return []

        if remember_port:
            self._used_ports += new_allocation

        return new_allocation
