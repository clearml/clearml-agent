import psutil


class TcpPorts(object):

    def __init__(self):
        self._used_ports = sorted([i.laddr.port for i in psutil.net_connections()])

    def check_tcp_port_available(self, port: int, remember_port: bool = True) -> bool:
        """
        return True if the port is available
        :param port: port number
        :param remember_port: if True add the port into the used ports list
        :return: True port is available
        """
        if port in self._used_ports:
            return False
        if remember_port:
            self._used_ports.append(port)
        return True

    def find_port_range(self, number_of_ports: int, remember_port: bool = True,
                        range_min: int = 10000, range_max: int = 60000) -> list:
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
